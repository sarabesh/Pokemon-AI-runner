"""Thin wrapper around PyBoy for driving Pokemon Red/Blue from an AI agent.

RAM addresses come from the pret/pokered disassembly (wram.asm).
"""
from pathlib import Path

from pyboy import PyBoy

BUTTONS = ["a", "b", "start", "select", "up", "down", "left", "right"]

# Pokemon Red/Blue (US) RAM map
ADDR = {
    "map_id": 0xD35E,
    "player_y": 0xD361,
    "player_x": 0xD362,
    "badges": 0xD356,      # bitfield, one bit per badge
    "party_count": 0xD163,
    "party_species": 0xD164,  # 6 bytes
    "battle_type": 0xD057,  # 0 = overworld, 1 = wild, 2 = trainer
    "money": 0xD347,        # 3 bytes BCD
}
PARTY_LEVEL = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
PARTY_HP = [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]  # 2 bytes, big-endian
PARTY_XP = [0xD179 + 0x2C * i for i in range(6)]  # 3 bytes, big-endian
PARTY_MOVES = [0xD173 + 0x2C * i for i in range(6)]  # 4 move ids each
PARTY_PP = [0xD188 + 0x2C * i for i in range(6)]  # 4 bytes each: low 6 bits PP, top 2 PP Ups
MOVES_TABLE = 0x0E * 0x4000  # ROM bank $0E:4000, 6 bytes per move, last byte = max PP
ENEMY_HP = 0xCFE6  # 2 bytes, big-endian (current enemy in battle)
TILEMAP = range(0xC3A0, 0xC3A0 + 20 * 18)  # wTileMap: the 20x18 tiles on screen
WINDOW_BORDER_TILE = 0x79  # top-left corner of every text box / menu
EVENT_FLAGS = (0xD747, 0xD747 + 320)  # wEventFlags: one bit per story event (2560 events)
OUTDOOR_MAX_MAP_ID = 0x24  # 0x00-0x0A towns/cities, 0x0C-0x24 Routes 1-25; higher ids are indoors


class PokemonEnv:
    def __init__(self, rom: str, headless: bool = True, speed: int = 0):
        if not Path(rom).exists():
            raise FileNotFoundError(f"ROM not found: {rom} (put your own dump in roms/)")
        self.pyboy = PyBoy(rom, window="null" if headless else "SDL2", sound_emulated=False)
        self.pyboy.set_emulation_speed(speed)  # 0 = unlimited
        self.mem = self.pyboy.memory
        rom_bytes = Path(rom).read_bytes()
        self.base_pp = [0] + [rom_bytes[MOVES_TABLE + (m - 1) * 6 + 5] for m in range(1, 166)]

    def step(self, button: str, hold_frames: int = 8, wait_frames: int = 24):
        """Press one button, then let the game settle. Returns the new state."""
        if button not in BUTTONS:
            raise ValueError(f"unknown button {button!r}, expected one of {BUTTONS}")
        self.pyboy.button(button, hold_frames)
        self.pyboy.tick(hold_frames + wait_frames, True)
        return self.state()

    def tick(self, frames: int = 1):
        self.pyboy.tick(frames, True)

    def screenshot(self):
        """160x144 PIL image of the current frame."""
        return self.pyboy.screen.image.copy()

    def _bcd(self, addr: int, n: int) -> int:
        out = 0
        for i in range(n):
            b = self.mem[addr + i]
            out = out * 100 + (b >> 4) * 10 + (b & 0xF)
        return out

    def context(self) -> str:
        """battle / window (dialogue or menu open) / overworld.

        Window borders are font tiles 0x79-0x7E in the screen tile buffer; map tiles never use them."""
        if self.mem[ADDR["battle_type"]] != 0:
            return "battle"
        if any(self.mem[a] == WINDOW_BORDER_TILE for a in TILEMAP):
            return "window"
        return "overworld"

    def _pp(self, count: int):
        """(current, max) PP summed over every move of the party."""
        cur = mx = 0
        for i in range(count):
            for j in range(4):
                move, b = self.mem[PARTY_MOVES[i] + j], self.mem[PARTY_PP[i] + j]
                if move and move < len(self.base_pp):
                    base = self.base_pp[move]
                    cur += b & 0x3F
                    mx += base + base // 5 * (b >> 6)
        return cur, mx

    def state(self) -> dict:
        count = min(self.mem[ADDR["party_count"]], 6)
        return {
            "context": self.context(),
            "map_id": self.mem[ADDR["map_id"]],
            "pos": (self.mem[ADDR["player_x"]], self.mem[ADDR["player_y"]]),
            "badges": bin(self.mem[ADDR["badges"]]).count("1"),
            "in_battle": self.mem[ADDR["battle_type"]] != 0,
            "enemy_hp": (self.mem[ENEMY_HP] << 8) | self.mem[ENEMY_HP + 1],
            "xp": sum((self.mem[a] << 16) | (self.mem[a + 1] << 8) | self.mem[a + 2]
                      for a in PARTY_XP[:count]),
            "pp": self._pp(count),
            "events": sum(bin(b).count("1") for b in self.mem[EVENT_FLAGS[0]:EVENT_FLAGS[1]]),
            "money": self._bcd(ADDR["money"], 3),
            "party": [
                {
                    "species_id": self.mem[ADDR["party_species"] + i],
                    "level": self.mem[PARTY_LEVEL[i]],
                    "hp": (self.mem[PARTY_HP[i]] << 8) | self.mem[PARTY_HP[i] + 1],
                }
                for i in range(count)
            ],
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            self.pyboy.save_state(f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.pyboy.load_state(f)

    def close(self):
        self.pyboy.stop(save=False)
