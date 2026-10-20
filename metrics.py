"""Progress metrics shared by every agent, so random / LLM / RL runs are comparable."""


class Tracker:
    def __init__(self):
        self.tiles = set()   # (map_id, x, y)
        self.maps = set()
        self.max_level_sum = 0
        self.max_badges = 0
        self.battles = 0
        self.battles_won = 0
        self.flees = 0
        self.max_xp = 0
        self._was_in_battle = False
        self._last_enemy_hp = None

    def update(self, state: dict):
        x, y = state["pos"]
        self.tiles.add((state["map_id"], x, y))
        self.maps.add(state["map_id"])
        self.max_level_sum = max(self.max_level_sum, sum(p["level"] for p in state["party"]))
        self.max_badges = max(self.max_badges, state["badges"])
        if state["in_battle"] and not self._was_in_battle:
            self.battles += 1
        # a battle that ends with the enemy at 0 HP was won (running away leaves it above 0)
        if self._was_in_battle and not state["in_battle"] and self._last_enemy_hp == 0:
            self.battles_won += 1
        elif (self._was_in_battle and not state["in_battle"] and self._last_enemy_hp
              and sum(p["hp"] for p in state["party"]) > 0):
            self.flees += 1  # ended with the enemy alive and us alive (also counts catches)
        if state["in_battle"]:
            self._last_enemy_hp = state.get("enemy_hp")
        self.max_xp = max(self.max_xp, state.get("xp", 0))
        self._was_in_battle = state["in_battle"]

    def summary(self) -> dict:
        return {
            "unique_tiles": len(self.tiles),
            "maps_visited": len(self.maps),
            "party_level_sum": self.max_level_sum,
            "badges": self.max_badges,
            "battles_started": self.battles,
            "battles_won": self.battles_won,
            "battles_fled": self.flees,
            "party_xp": self.max_xp,
        }
