# utils.py
import math
import difflib
from enum import Enum
from typing import List, Tuple, Dict, Optional
from config import RAW_NEUTRAL_WR, RAW_NEUTRAL_KDA, RAW_NEUTRAL_PR, SYNERGY_SUPPORT_BOOST

class SynergyStatus(Enum):
    OK = "OK"
    NO_TOKEN = "NO_TOKEN"
    FAILED = "FAILED"
    NO_DATA = "NO_DATA"
    PARTIAL_DATA = "PARTIAL_DATA"

def convert_id64_to_id32(steam_id: int) -> int:
    return steam_id - 76561197960265728 if steam_id > 76561197960265728 else steam_id

def is_win(player_slot: int, radiant_win: bool) -> bool:
    # 0-127 - Radiant, 128-255 - Dire
    return (player_slot < 128) == radiant_win

def calc_kda(kills: float, deaths: float, assists: float) -> float:
    return (kills + assists) / max(1.0, deaths)

def smooth_winrate(wins: float, games: float, prior_games: float, prior_wr: float) -> float:
    if games <= 0: return prior_wr
    return (wins + prior_games * prior_wr) / (games + prior_games)

def calculate_decay_weight(start_time: float, current_time: float, half_life: float) -> float:
    days_ago = max(0.0, (current_time - start_time) / 86400.0)
    return math.pow(0.5, days_ago / half_life)

def synergy_role_weight(my_role: int, ally_role: Optional[int], boost: float = SYNERGY_SUPPORT_BOOST) -> float:
    """Вес синергии с союзником по позициям: керри/мид <-> саппорты важнее остальных связок."""
    if ally_role and ((my_role in (1, 2) and ally_role in (4, 5)) or (my_role in (4, 5) and ally_role in (1, 2))):
        return boost
    return 1.0

def candidate_wr_vs_enemy(enemy_wins: float, games: float) -> float:
    """Винрейт кандидата = 1 - (победы врага над кандидатом / матчи)."""
    if games <= 0: return 0.5
    return 1.0 - (enemy_wins / games)

def calculate_advantage(target_wr: float, base_wr: float) -> float:
    return target_wr - base_wr

def sigmoid_scaled(advantage: float, scale: float) -> float:
    if scale == 0: return 0.5
    x = max(-500.0, min(500.0, advantage / scale))
    return 1.0 / (1.0 + math.exp(-x))

def calculate_absolute_form(raw_wr: float, raw_kda: float, raw_pr_norm: float,
                            eff_games: float, shrink_k: float, form_gain: float) -> float:
    """Абсолютная оценка формы с шринкиджем к 0.5 по эффективному числу игр."""
    neutral_score = (0.6 * RAW_NEUTRAL_WR) + (0.3 * min(RAW_NEUTRAL_KDA / 5.0, 1.0)) + (0.1 * RAW_NEUTRAL_PR)
    actual_score = (0.6 * raw_wr) + (0.3 * min(raw_kda / 5.0, 1.0)) + (0.1 * raw_pr_norm)
    diff = actual_score - neutral_score
    shrinkage = eff_games / (eff_games + shrink_k) if (eff_games + shrink_k) > 0 else 0
    return max(0.0, min(1.0, 0.5 + diff * form_gain * shrinkage))

def build_form_scores(matches: List[dict], current_time: float, half_life: float,
                      prior_games: float, shrink_k: float, form_gain: float) -> Dict[int, dict]:
    """Глобальная нормировка весов (средний вес выборки = 1) и абсолютная форма героев."""
    if not matches:
        return {}
    weights = [calculate_decay_weight(m['start_time'], current_time, half_life) for m in matches]
    mean_w = sum(weights) / len(weights)
    if mean_w == 0: mean_w = 1.0
    norm_weights = [w / mean_w for w in weights]

    hero_stats: Dict[int, dict] = {}
    for i, m in enumerate(matches):
        hid = m.get('hero_id', 0)
        if hid == 0: continue
        if hid not in hero_stats:
            hero_stats[hid] = {'w_games': 0.0, 'w_wins': 0.0, 'w_k': 0.0, 'w_d': 0.0, 'w_a': 0.0,
                               'raw_games': 0, 'raw_wins': 0, 'raw_k': 0, 'raw_d': 0, 'raw_a': 0}
        s = hero_stats[hid]
        w = norm_weights[i]
        win = is_win(m['player_slot'], m['radiant_win'])
        s['w_games'] += w
        if win: s['w_wins'] += w
        s['w_k'] += m.get('kills', 0) * w
        s['w_d'] += m.get('deaths', 0) * w
        s['w_a'] += m.get('assists', 0) * w
        s['raw_games'] += 1
        if win: s['raw_wins'] += 1
        s['raw_k'] += m.get('kills', 0)
        s['raw_d'] += m.get('deaths', 0)
        s['raw_a'] += m.get('assists', 0)

    max_eff_games = max((s['w_games'] for s in hero_stats.values()), default=1.0)
    forms = {}
    for hid, s in hero_stats.items():
        eff_games = s['w_games']
        wr_smooth = smooth_winrate(s['w_wins'], eff_games, prior_games, RAW_NEUTRAL_WR)
        kda_w = calc_kda(s['w_k'], s['w_d'], s['w_a'])
        pr_norm = min(1.0, eff_games / max_eff_games)
        forms[hid] = {
            'score': calculate_absolute_form(wr_smooth, kda_w, pr_norm, eff_games, shrink_k, form_gain),
            'eff_games': eff_games,
            'games': s['raw_games'],
            'wr': s['raw_wins'] / s['raw_games'] if s['raw_games'] > 0 else 0,
            'kda': calc_kda(s['raw_k'], s['raw_d'], s['raw_a']),
        }
    return forms

def parse_matchup_response(data: dict) -> Tuple[SynergyStatus, str, dict, dict]:
    """
    Разбор ответа Stratz heroStats.matchUp. Возвращает (статус, пояснение, with{heroId2: запись}, vs{heroId2: запись}).
    Структура проверена в Explorer: matchUp = список из одного элемента, внутри with[...] и vs[...].
    Принимает и dict, и list на узле matchUp.
    """
    if not data:
        return SynergyStatus.FAILED, "Empty Response", {}, {}
    if "errors" in data and not data.get("data"):
        try:
            msg = str(data["errors"][0].get("message", ""))[:160]
        except Exception:
            msg = ""
        return SynergyStatus.FAILED, f"GraphQL Errors: {msg}".strip(), {}, {}
    try:
        matchup = data.get("data", {}).get("heroStats", {}).get("matchUp")
        if not matchup:
            return SynergyStatus.NO_DATA, "No matchUp field", {}, {}
        items = matchup if isinstance(matchup, list) else [matchup]
        with_list, vs_list = [], []
        for item in items:
            with_list.extend(item.get("with") or [])
            vs_list.extend(item.get("vs") or [])
        with_d = {x['heroId2']: x for x in with_list}
        vs_d = {x['heroId2']: x for x in vs_list}
        if not with_d and not vs_d:
            return SynergyStatus.NO_DATA, "Empty 'with'/'vs' arrays", {}, {}
        return SynergyStatus.OK, "Success", with_d, vs_d
    except Exception as e:
        return SynergyStatus.FAILED, f"Parse Error: {e}", {}, {}

def parse_synergy_response(data: dict) -> Tuple[SynergyStatus, str, dict]:
    """Только блок with (синергия)."""
    st, detail, with_d, _ = parse_matchup_response(data)
    if st == SynergyStatus.OK and not with_d:
        return SynergyStatus.NO_DATA, "Empty 'with' array", {}
    return st, detail, with_d

def aggregate_synergy_status(per_ally: Dict[str, Tuple[SynergyStatus, str]]) -> Tuple[SynergyStatus, str, List[str]]:
    """
    Общий статус синергии по всем союзникам.
    Возвращает (статус, пояснение, список союзников без данных).
      все OK                         -> OK
      есть и OK, и не-OK             -> PARTIAL_DATA (в пояснении причины по проблемным)
      ни одного OK:
        нет токена                   -> NO_TOKEN
        есть FAILED                  -> FAILED
        иначе (все NO_DATA)          -> NO_DATA
    """
    if not per_ally:
        return SynergyStatus.OK, "", []
    bad = {n: sd for n, sd in per_ally.items() if sd[0] != SynergyStatus.OK}
    if not bad:
        return SynergyStatus.OK, "Success", []
    missing = list(bad)
    if len(bad) < len(per_ally):
        detail = "; ".join(f"{n}: {d}" for n, (s, d) in bad.items())
        return SynergyStatus.PARTIAL_DATA, detail, missing
    statuses = [s for s, _ in bad.values()]
    if SynergyStatus.NO_TOKEN in statuses:
        return SynergyStatus.NO_TOKEN, "нет токена STRATZ_TOKEN", missing
    for s, d in bad.values():
        if s == SynergyStatus.FAILED:
            return SynergyStatus.FAILED, d, missing
    return SynergyStatus.NO_DATA, "нет данных по связкам", missing

def search_hero(query: str, search_map: Dict[str, int], heroes: Dict[int, any]) -> Tuple[Optional[any], List[any]]:
    """Поиск героя: точное совпадение/алиас -> fuzzy (не для строк до 2 символов)."""
    q = query.strip().lower()
    if not q: return None, []
    if q in search_map:
        return heroes[search_map[q]], []
    if len(q) <= 2:
        return None, []
    matches = difflib.get_close_matches(q, search_map.keys(), n=3, cutoff=0.6)
    suggested, seen = [], set()
    for m in matches:
        hid = search_map[m]
        if hid not in seen:
            seen.add(hid)
            suggested.append(heroes[hid])
    return None, suggested

def std_dev(values: List[float]) -> float:
    if len(values) < 2: return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

def contribution_report(candidates) -> Tuple[float, float, float, float, float, float]:
    """(std формы, std драфта, доля формы, доля драфта, среднее |преим.|, макс |преим.|)."""
    std_f = std_dev([c.form_score for c in candidates])
    std_d = std_dev([c.draft_score for c in candidates])
    wf, wd = 0.4 * std_f, 0.6 * std_d
    total = wf + wd
    fract_f = wf / total if total > 0 else 0.0
    fract_d = wd / total if total > 0 else 0.0
    advs = [abs(m.advantage) for c in candidates for m in c.matchups]
    return std_f, std_d, fract_f, fract_d, (sum(advs) / len(advs) if advs else 0.0), (max(advs) if advs else 0.0)
