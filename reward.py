"""Reward function over RAM state, built from the reward ideas in controls.json.

Kept separate from the env so the parent LLM can rewrite it between training runs.
Each component is logged on its own, so it's visible which one the policy is chasing.

Versions:
  v1  exploration-first (run "first"): tiles + maps dominate, fighting barely pays.
  v2  fighting and story progress pay about as well as exploring; outdoor travel
      (towns/routes) pays more than wandering into buildings. Result: grinds Route 1.
  v5  v4 with battles paying less (XP /3, cap level 10), exploration paying 2-6x more,
      story events 100, badges 300, and
      -1 for any overworld press that changes nothing (e.g. walking into a wall).
  v4  v3 + fleeing costs -3, XP linear but capped by level (no endless grinding), story
      events 40 and they reset exploration memory so backtracking pays.
  v3  no per-fight rewards; Pokemon XP on a diminishing sqrt curve, story events x3,
      a small per-press penalty after 300 presses without progress, and another while
      party PP is below 25% (go heal).
"""
from env import OUTDOOR_MAX_MAP_ID

PRESETS = {
    "v1": {
        "new_tile": 0.1,     # first visit to (map_id, x, y) this episode
        "new_map": 2.0,      # first visit to a map_id this episode
        "level_up": 1.0,     # per party level gained
        "badge": 20.0,       # per badge
        "damage": 0.05,      # per enemy HP removed in battle
        "blackout": -5.0,    # whole party fainted
    },
    "v2": {
        "new_tile": 0.1,             # indoor tile
        "new_tile_outdoor": 0.2,     # tile in a town or on a route
        "new_map": 1.0,              # indoor map (houses, labs, marts)
        "new_map_outdoor": 5.0,      # new town or route = actually travelling somewhere
        "level_up": 5.0,
        "badge": 20.0,
        "damage": 0.3,               # a ~15 HP wild Pokemon is worth ~4.5
        "enemy_fainted": 5.0,        # battle won (per enemy Pokemon knocked out)
        "event": 10.0,               # per new story event flag set (parcel, Pokedex, ...)
        "blackout": -5.0,
    },
    # v2 grinding fix: run "v2" sat in Route 1 grass (~35 battles, 80 tiles per episode) because
    # damage + knockouts paid ~85% of its reward. v3 drops all per-fight rewards; fighting only
    # pays through XP, on a square-root curve so each extra fight in the same area pays less.
    # Story events are the main prize.
    "v3": {
        "new_tile": 0.1,
        "new_tile_outdoor": 0.2,
        "new_map": 1.0,
        "new_map_outdoor": 5.0,
        "xp": 2.0,                   # per unit of sqrt(total party XP): a 23-XP Pidgey pays
                                     # ~1.6 at L6 but ~0.85 at L11; grinding L6->L11 ~ +26 total
        "event": 30.0,               # story progress is the main objective
        "badge": 50.0,
        "blackout": -10.0,           # losing costs more, so it learns to fight to win
        "stuck": -0.05,              # per press once `stuck_after` presses pass with no progress
        "stuck_after": 300,          # (new tile, new map, story event or badge); a long trainer
                                     # battle is ~150 presses, so single fights stay unpunished
        "low_pp": -0.05,             # per press while party PP is below `low_pp_below` of max:
        "low_pp_below": 0.25,        # pushes it to rest (mom in Pallet / Pokemon Center) before
                                     # it runs dry and is left with Struggle
    },
    # v3 result: ran from ~all battles (XP ~1.6 per win wasn't worth the presses), stalled at
    # Viridian (road north opens only after carrying Oak's Parcel back to Pallet, and the trip
    # back paid nothing while the stuck penalty punished it), policy stayed near random.
    "v4": {
        "new_tile": 0.1,
        "new_tile_outdoor": 0.2,
        "new_map": 1.0,
        "new_map_outdoor": 5.0,
        "xp": 0.15,                  # per XP point, linear: a 23-XP Pidgey ~ +3.5 ...
        "xp_level_cap": 12,          # ... but only while the strongest Pokemon is below the cap
        "xp_cap_per_badge": 6,       # (12 now, +6 per badge), so grinding stops paying
        "flee": -3.0,                # wild battle ended with the enemy alive and nothing caught
        "event": 40.0,
        "reset_explore_on_event": 1, # after each story event, visited tiles/maps pay again, so
                                     # going back (parcel -> Pallet) is rewarded like exploring
        "badge": 50.0,
        "blackout": -10.0,
        "stuck": -0.05,
        "stuck_after": 500,          # a bit more slack: menus + battles + healing take presses
        "low_pp": -0.05,
        "low_pp_below": 0.25,
    },
    # v4 result: best explorer so far (~790 tiles) and fights to win (26 won vs 3 fled), but it
    # farms Route 1 up to the level cap (XP ~25% of positive reward, stuck penalty -43) and
    # gets the parcel without delivering it. v5 pays battles less and new ground more, and
    # punishes presses that change nothing (walking into walls, A/B at nothing).
    "v5": {
        "new_tile": 0.2,             # x2
        "new_tile_outdoor": 0.6,     # x3
        "new_map": 3.0,              # x3
        "new_map_outdoor": 30.0,     # x6: reaching a new town/route is the big exploration prize
        "xp": 0.05,                  # /3: a 23-XP Pidgey ~ +1.2
        "xp_level_cap": 10,          # grinding stops paying at level 10 (was 12)
        "xp_cap_per_badge": 6,
        "flee": -1.0,                # softer: it no longer needs a push into fights
        "event": 100.0,              # x2.5: story progress (parcel, Pokedex, ...) is the goal
        "reset_explore_on_event": 1,
        "badge": 300.0,              # x6: the biggest single prize
        "blackout": -10.0,
        "no_effect": -1.0,           # overworld press that moved nothing: wall bump, A/B at air
        "stuck": -0.05,
        "stuck_after": 300,          # back to 300: farming already-walked grass hits it sooner
        "low_pp": -0.05,
        "low_pp_below": 0.25,
    },
}
WEIGHTS = PRESETS["v1"]


class Reward:
    def __init__(self, weights=None, version="v1"):
        self.w = {**PRESETS[version], **(weights or {})}
        self.stuck_after = self.w.pop("stuck_after", None)  # thresholds, not reward weights
        self.low_pp_below = self.w.pop("low_pp_below", None)
        self.xp_level_cap = self.w.pop("xp_level_cap", None)  # None = v3 sqrt XP
        self.xp_cap_per_badge = self.w.pop("xp_cap_per_badge", 0)
        self.reset_explore = bool(self.w.pop("reset_explore_on_event", 0))
        self.reset()

    def reset(self):
        self.tiles = set()
        self.maps = set()
        self.max_events = None
        self.max_xp_root = None
        self.max_xp = None
        self.battle_enemy_hp = None  # enemy HP on the latest in-battle step
        self.last_parts = {}
        self.stale = 0  # presses since the last progress
        self.totals = {k: 0.0 for k in self.w}

    def __call__(self, prev: dict, state: dict, prev_enemy_hp: int, enemy_hp: int) -> float:
        parts = {}
        outdoor = state["map_id"] <= OUTDOOR_MAX_MAP_ID
        split = "new_tile_outdoor" in self.w  # v2 pays outdoor exploration separately

        x, y = state["pos"]
        tile = (state["map_id"], x, y)
        if tile not in self.tiles:
            self.tiles.add(tile)
            parts["new_tile_outdoor" if split and outdoor else "new_tile"] = 1
        if state["map_id"] not in self.maps:
            self.maps.add(state["map_id"])
            parts["new_map_outdoor" if split and outdoor else "new_map"] = 1

        levels = sum(p["level"] for p in state["party"])
        prev_levels = sum(p["level"] for p in prev["party"])
        # level 0 shows up for a frame while a Pokemon is being added; ignore drops and that blip
        if prev["party"] and all(p["level"] for p in state["party"]) and levels > prev_levels:
            parts["level_up"] = levels - prev_levels
        if state["badges"] > prev["badges"]:
            parts["badge"] = state["badges"] - prev["badges"]
        if state["in_battle"] and prev["in_battle"] and enemy_hp < prev_enemy_hp:
            parts["damage"] = prev_enemy_hp - enemy_hp
            if enemy_hp == 0:
                parts["enemy_fainted"] = 1

        # XP above the episode's high-water mark (depositing a Pokemon lowers party XP; that
        # must not pay again when it comes back)
        if self.xp_level_cap is None:  # v3: sqrt curve
            xp_root = state.get("xp", 0) ** 0.5
            if self.max_xp_root is None:
                self.max_xp_root = xp_root
            elif xp_root > self.max_xp_root:
                parts["xp"] = xp_root - self.max_xp_root
                self.max_xp_root = xp_root
        else:  # v4: linear, only below the level cap
            xp = state.get("xp", 0)
            if self.max_xp is None:
                self.max_xp = xp
            elif xp > self.max_xp:
                cap = self.xp_level_cap + self.xp_cap_per_badge * state["badges"]
                if max((p["level"] for p in prev["party"]), default=0) < cap:
                    parts["xp"] = xp - self.max_xp
                self.max_xp = xp

        # fleeing: a wild battle that ends with the enemy still standing, our party alive and
        # no new party member (a catch). Trainer battles can't be fled.
        if state["in_battle"]:
            self.battle_enemy_hp = enemy_hp
        elif prev["in_battle"]:
            alive = sum(p["hp"] for p in state["party"]) > 0
            caught = len(state["party"]) > len(prev["party"])
            if self.battle_enemy_hp and alive and not caught:
                parts["flee"] = 1
            self.battle_enemy_hp = None

        # story progress: count new flags above the episode's high-water mark, so a flag
        # that gets cleared and set again can't be farmed
        if self.max_events is None:
            self.max_events = state["events"]
        elif state["events"] > self.max_events:
            parts["event"] = state["events"] - self.max_events
            self.max_events = state["events"]
            if self.reset_explore:  # the world is "new" again after story progress
                self.tiles = {tile}
                self.maps = {state["map_id"]}

        # wasted press: walking and standing in the overworld with nothing changing - bumping
        # into a wall, A/B facing nothing. Opening a dialogue, a battle, a warp or any step
        # changes position, map or context, so those aren't punished.
        if (prev["context"] == "overworld" and state["context"] == "overworld"
                and prev["pos"] == state["pos"] and prev["map_id"] == state["map_id"]):
            parts["no_effect"] = 1

        # stagnation: grinding one patch of grass or pacing a room earns nothing new; past the
        # threshold every extra press costs a little
        progress = {"new_tile", "new_tile_outdoor", "new_map", "new_map_outdoor", "event", "badge"}
        self.stale = 0 if progress & parts.keys() else self.stale + 1
        if self.stuck_after is not None and self.stale > self.stuck_after:
            parts["stuck"] = 1

        pp, pp_max = state.get("pp", (0, 0))
        if self.low_pp_below is not None and pp_max and pp / pp_max < self.low_pp_below:
            parts["low_pp"] = 1

        hp = sum(p["hp"] for p in state["party"])
        prev_hp = sum(p["hp"] for p in prev["party"])
        if state["party"] and hp == 0 and prev_hp > 0:
            parts["blackout"] = 1

        self.last_parts = parts
        total = 0.0
        for k, v in parts.items():
            if k not in self.w:  # e.g. v1 has no enemy_fainted / event components
                continue
            r = self.w[k] * v
            self.totals[k] += r
            total += r
        return total
