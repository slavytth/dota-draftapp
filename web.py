# web.py — сайт + Telegram-бот в одном файле для Render
import json
import os
import threading
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, WebAppInfo

import config
from analysis import DotaAnalyzer, select_recommendations
from api import OpenDotaClient, StratzClient, ApiUnavailableError, PrivateProfileError, EmptyHistoryError
from utils import convert_id64_to_id32, contribution_report

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = threading.Lock()
ANALYZER = None
PORT = 8000


def run_bot():
    """Запускает Telegram-бота в фоновом режиме."""
    token = os.environ.get("BOT_TOKEN")
    if not token:
        print("⚠️ BOT_TOKEN не найден в переменной окружения! Бот не запущен.")
        return

    web_app_url = "https://dota-draftapp-3.onrender.com"
    bot = telebot.TeleBot(token)

    @bot.message_handler(commands=['start'])
    def start(message):
        markup = ReplyKeyboardMarkup(resize_keyboard=True)
        web_app = WebAppInfo(url=web_app_url)
        btn = KeyboardButton(text="🎮 Открыть Драфт-помощника", web_app=web_app)
        markup.add(btn)

        bot.send_message(
            message.chat.id, 
            "Привет! Нажми на кнопку ниже, чтобы выбрать героев:", 
            reply_markup=markup
        )

    print("🤖 Telegram-бот успешно запущен!")
    try:
        bot.infinity_polling()
    except Exception as e:
        print(f"❌ Ошибка в работе бота: {e}")


def heroes_payload() -> dict:
    heroes = []
    for hid, h in ANALYZER.heroes.items():
        heroes.append({
            "id": hid,
            "en": ANALYZER.heroes_data[hid]["localized_name"],
            "ru": h.loc_name,
            "slug": h.name.replace("npc_dota_hero_", ""),
            "roles": h.roles,
        })
    heroes.sort(key=lambda x: x["en"])
    return {
        "heroes": heroes,
        "config": {"strong_adv": config.STRONG_COUNTER_ADV,
                   "excellent_form": config.EXCELLENT_FORM_THRESHOLD,
                   "min_matchup_games": config.MIN_MATCHUP_GAMES},
        "has_stratz": bool(ANALYZER.stratz.token),
    }


def _entries(items):
    ids, roles = [], {}
    for x in items or []:
        i = int(x["id"] if isinstance(x, dict) else x)
        if i in ids:
            continue
        ids.append(i)
        r = x.get("role") if isinstance(x, dict) else None
        if r:
            if int(r) not in (1, 2, 3, 4, 5):
                raise ValueError("Позиция героя должна быть от 1 до 5")
            roles[i] = int(r)
    return ids, roles


def recommend(p: dict) -> dict:
    acc_id = convert_id64_to_id32(int(p["id"]))
    role = int(p["role"])
    if role not in (1, 2, 3, 4, 5):
        raise ValueError("Роль должна быть от 1 до 5")
    enemy_ids, _ = _entries(p.get("enemies"))
    enemy_ids = enemy_ids[:5]
    ally_ids, ally_roles = _entries(p.get("allies"))
    ally_ids = [i for i in ally_ids if i not in enemy_ids][:4]
    enemies = [ANALYZER.heroes[i] for i in enemy_ids]
    allies = [ANALYZER.heroes[i] for i in ally_ids]
    matches = max(50, min(100, int(p.get("matches", 100))))
    min_games = max(1, int(p.get("min_games", 3)))
    with_meta = bool(p.get("with_meta", True))

    with LOCK:
        res = ANALYZER.get_recommendations(acc_id, role, enemies, allies, matches, min_games, False, ally_roles)
    picks = select_recommendations(res.candidates, n_pool=2 if with_meta else 3, n_total=3, min_games=min_games)
    cands = []
    for kind, c in picks:
        d = asdict(c)
        d["kind"] = kind
        d["name"] = c.hero.loc_name
        d["en"] = ANALYZER.heroes_data[c.hero.id]["localized_name"]
        d["slug"] = c.hero.name.replace("npc_dota_hero_", "")
        cands.append(d)
    pool_found = sum(1 for k, _ in picks if k == "pool")
    if pool_found < (2 if with_meta else 3):
        res.warnings.append(f"В твоём пуле на этой позиции подходящих героев: {pool_found}. Остальные места заняты мета-пиками.")
    sf, sd, ff, fd, _, _ = contribution_report(res.candidates)
    return {
        "contribution": {"std_form": sf, "std_draft": sd, "form_share": ff, "draft_share": fd},
        "candidates": cands,
        "total_candidates": len(res.candidates),
        "synergy_status": res.synergy_status.value,
        "synergy_detail": res.synergy_detail,
        "counter_source": res.counter_source,
        "matches_used": res.matches_used,
        "matches_requested": res.matches_requested,
        "warnings": res.warnings,
        "account_id": acc_id,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/heroes":
            self._json(200, heroes_payload())
        else:
            self._send(404, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/recommend":
            return self._send(404, b"Not found", "text/plain")
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, recommend(payload))
        except (PrivateProfileError, EmptyHistoryError, ApiUnavailableError) as e:
            hint = " Включи в Dota 2 «Expose Public Match Data» (Настройки → Социальные)." \
                if isinstance(e, (PrivateProfileError, EmptyHistoryError)) else ""
            self._json(200, {"error": f"{e}{hint}"})
        except (ValueError, KeyError, TypeError) as e:
            self._json(400, {"error": f"Некорректный запрос: {e}"})
        except Exception as e:
            self._json(500, {"error": f"Внутренняя ошибка: {e}"})

    def log_message(self, *args):
        pass


def main():
    global ANALYZER
    print("Загружаю список героев с OpenDota...")
    try:
        ANALYZER = DotaAnalyzer(OpenDotaClient(), StratzClient())
    except ApiUnavailableError as e:
        print(f"❌ Не удалось связаться с OpenDota: {e}")
        return

    # Запускаем бота в отдельном фоновом потоке
    threading.Thread(target=run_bot, daemon=True).start()

    port = int(os.environ.get("PORT", 8000))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    
    url = f"http://127.0.0.1:{port}"
    print(f"Готово. Сервер запущен на порту {port}")
    
    if "RENDER" not in os.environ:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")


if __name__ == "__main__":
    main()