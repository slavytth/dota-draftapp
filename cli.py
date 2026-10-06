# cli.py
import argparse
from typing import List, Optional
from analysis import DotaAnalyzer, HeroInfo, CandidateResult, RecommendationResult
from api import OpenDotaClient, StratzClient, ApiUnavailableError, PrivateProfileError, EmptyHistoryError
from utils import convert_id64_to_id32, contribution_report, SynergyStatus
from config import MIN_MATCHUP_GAMES, EXCELLENT_FORM_THRESHOLD, STRONG_COUNTER_ADV

def resolve_heroes_interactive(names_str: str, analyzer: DotaAnalyzer, max_count: int, ignore_list: List[int]) -> List[HeroInfo]:
    result: List[HeroInfo] = []
    if not names_str: return result
    names = [n.strip() for n in names_str.split(',') if n.strip()]
    for name in names:
        if len(result) >= max_count:
            print(f"⚠️ Достигнут лимит ({max_count}). Остальные проигнорированы.")
            break
        exact, suggs = analyzer.resolve_hero(name)
        chosen = exact
        if not chosen and suggs:
            print(f"\nГерой '{name}' неоднозначен. Варианты:")
            for i, s in enumerate(suggs, 1): print(f"  {i}. {s.loc_name}")
            try:
                idx = input("Номер (Enter пропустить): ")
                if idx.isdigit() and 1 <= int(idx) <= len(suggs): chosen = suggs[int(idx) - 1]
            except EOFError: pass
        if not chosen:
            print(f"❌ Герой '{name}' не найден и пропущен.")
            continue
        if chosen.id in ignore_list or any(h.id == chosen.id for h in result):
            print(f"⚠️ Герой {chosen.loc_name} уже в пике/бане. Пропущен.")
            continue
        result.append(chosen)
    return result

def check_range(val, min_v, max_v, name):
    v = int(val)
    if not (min_v <= v <= max_v): raise argparse.ArgumentTypeError(f"{name} должен быть {min_v}-{max_v}")
    return v

def synergy_line(c: CandidateResult, res: RecommendationResult, allies_given: bool) -> Optional[str]:
    """Строка про синергию. Если союзники указаны, строка есть ВСЕГДА (значения / нет данных / отключена)."""
    if not allies_given:
        return None
    if res.synergy_status in (SynergyStatus.NO_TOKEN, SynergyStatus.FAILED):
        return f"🤝 Синергия: отключена — {res.synergy_detail}"
    if c.synergies:
        text = ", ".join(f"{s.ally_name} ({s.advantage * 100:+.1f} п.п., {s.games} игр)" for s in c.synergies)
        if c.synergy_missing:
            text += f" (нет данных по: {', '.join(c.synergy_missing)})"
        return f"🤝 Синергия: {text}"
    return f"🤝 Синергия: нет данных по: {', '.join(c.synergy_missing)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int)
    parser.add_argument("--role", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--enemies", type=str, default="")
    parser.add_argument("--allies", type=str, default="")
    parser.add_argument("--matches", type=lambda x: check_range(x, 50, 100, "matches"), default=100)
    parser.add_argument("--top", type=lambda x: check_range(x, 1, 4, "top"), default=3)
    parser.add_argument("--min-games", type=lambda x: check_range(x, 1, 999, "min-games"), default=3)
    parser.add_argument("--only-played", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Показать диагностику вкладов шкал")
    args = parser.parse_args()

    try:
        analyzer = DotaAnalyzer(OpenDotaClient(), StratzClient())
    except ApiUnavailableError as e:
        print(f"\n❌ Ошибка сети при инициализации: {e}")
        return

    acc_id = args.id
    while not acc_id:
        try: acc_id = int(input("Steam ID (32/64): "))
        except ValueError: pass
    acc_id = convert_id64_to_id32(acc_id)

    role = args.role
    while not role or role not in [1, 2, 3, 4, 5]:
        try: role = int(input("Роль (1-5): "))
        except ValueError: pass

    enemies = resolve_heroes_interactive(args.enemies or input("Враги через запятую: "), analyzer, 5, [])
    allies = resolve_heroes_interactive(args.allies or input("Союзники через запятую: "), analyzer, 4, [e.id for e in enemies])

    print(f"\nАнализ (ID: {acc_id}, Роль: {role}, Запрошено матчей: {args.matches})...")
    try:
        res = analyzer.get_recommendations(acc_id, role, enemies, allies, args.matches, args.min_games, args.only_played)
    except (PrivateProfileError, EmptyHistoryError, ApiUnavailableError) as e:
        print(f"\n❌ Ошибка: {e}\n(Если профиль пуст, убедитесь, что включен 'Expose Public Match Data')")
        return

    if not res.candidates:
        print("\nНет подходящих кандидатов.")
        return

    for w in res.warnings: print(f"⚠️ {w}")
    if enemies: print(f"ℹ️ Источник контрпиков: {res.counter_source}")

    print(f"\n🏆 ТОП-{args.top} КАНДИДАТОВ:")
    for i, c in enumerate(res.candidates[:args.top], 1):
        print(f"\n{i}. \033[36;1m{c.hero.loc_name}\033[0m (Итог: {c.final_score:.2f})")
        if c.form_games > 0:
            print(f"   📊 Форма: {c.form_games} игр, WR {int(c.form_wr * 100)}%, KDA {c.form_kda:.1f} (Балл: {c.form_score:.2f})")
        else:
            print("   📊 Форма: Нет недавней практики")

        if c.expected_wr is not None:
            print(f"   🎯 Ожидаемый WR против этих врагов: {c.expected_wr * 100:.1f}% (средний WR героя {c.base_wr * 100:.1f}%)")
        c.matchups.sort(key=lambda x: x.advantage, reverse=True)
        good = [f"{m.enemy_name} (WR {m.raw_wr * 100:.0f}%, {m.advantage * 100:+.1f} п.п. после поправки, {m.games} игр)" for m in c.matchups if m.advantage >= STRONG_COUNTER_ADV]
        bad = [f"{m.enemy_name} (WR {m.raw_wr * 100:.0f}%, {m.advantage * 100:+.1f} п.п. после поправки, {m.games} игр)" for m in c.matchups if m.advantage <= -STRONG_COUNTER_ADV]
        if good: print(f"   🟢 Силен против: {', '.join(good)}")
        if bad: print(f"   🔴 Слаб против: {', '.join(bad)}")

        line = synergy_line(c, res, bool(allies))
        if line: print(f"   {line}")

        for m in c.matchups:
            if m.warnings: print(f"   ⚠️ {m.enemy_name}: {', '.join(m.warnings)}")
        for w in c.warnings: print(f"   ⚠️ {w}")

        exp = []
        if c.form_games > 0 and c.form_score >= EXCELLENT_FORM_THRESHOLD:
            exp.append("Отличная личная форма на герое.")
        if any(m.advantage >= STRONG_COUNTER_ADV and m.games >= MIN_MATCHUP_GAMES for m in c.matchups):
            exp.append("Сильный контрпик текущему драфту.")
        if exp: print(f"   💡 {' '.join(exp)}")

    if args.debug:
        sf, sd, wf, wd, avg_a, max_a = contribution_report(res.candidates)
        print("\n[DEBUG] ДИАГНОСТИКА ШКАЛ")
        print(f"Std Dev: Форма = {sf:.3f} | Драфт = {sd:.3f}")
        print(f"Вклад в разброс (вес 40/60): Форма = {wf * 100:.1f}% | Драфт = {wd * 100:.1f}%")
        print(f"Преимущество контрпика: среднее = {avg_a * 100:.1f} п.п. | максимальное = {max_a * 100:.1f} п.п.")
        print("Подсказка: для баланса 40/60 меняйте COUNTER_SCALE/SYNERGY_SCALE или FORM_GAIN в config.py.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
