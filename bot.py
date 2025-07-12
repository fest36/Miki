import logging
import re
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from langchain.prompts import PromptTemplate
import aiosqlite
import aiohttp
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from typing import Optional

# Настройка логирования для отслеживания работы бота
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка конфигурации из файла .env
load_dotenv()

# Пользовательский ввод: Вставьте ваш Telegram Bot Token, полученный от @BotFather
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Пользовательский ввод: Вставьте ваш OpenRouter API ключ
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Пользовательский ввод: Вставьте ваш Google Custom Search API ключ
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Пользовательский ввод: Вставьте ваш Google Custom Search Engine ID
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Имя бота и его вариации
BOT_NAMES = {"miki", "Miki", "мики", "Мики"}

# Модель для валидации пользовательского ввода
class UserInput(BaseModel):
    text: str
    user_id: str
    language: str = "ru"
    user_level: str = "beginner"  # По умолчанию начальный уровень для простых объяснений

# Инициализация шаблона для запросов к OpenRouter
PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["user_input", "context", "language", "user_level"],
    template=(
        "Ты ИТ-ассистент по имени Miki, созданный для помощи с программированием, настройкой программ и другими ИТ-задачами. "
        "Твой стиль общения — дружелюбный, простой и понятный, как будто объясняешь другу. "
        "Пользователь спросил: {user_input}. Контекст беседы: {context}. Язык ответа: {language}. "
        "Уровень знаний пользователя: {user_level} (новичок, продвинутый или эксперт). "
        "Для новичков объясняй всё максимально просто, шаг за шагом, без сложных слов. "
        "Используй Markdown для форматирования: списки, таблицы, кодовые блоки (```) для примеров кода. "
        "Если запрос связан с поиском, укажи, что можешь найти информацию в интернете. "
        "Если запрос неясен, задай уточняющий вопрос (например, 'Какую программу ты хочешь написать?'). "
        "Если запрос не про ИТ, вежливо скажи, что ты лучше разбираешься в ИТ, но попробуй помочь или предложи поиск."
    )
)

# Проверка безопасности пользовательского ввода на наличие опасных команд
def is_safe_input(text: str) -> bool:
    dangerous_patterns = [r"\bexec\b", r"\beval\b", r"\bimport os\b", r"\bimport sys\b", r"\brm -rf\b", r"\b__import__\b"]
    return not any(re.search(pattern, text, re.IGNORECASE) for pattern in dangerous_patterns)

# Асинхронный запрос к OpenRouter API
async def query_openrouter(prompt: str, model: str = "deepseek/deepseek-chat-v3-0324", retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
                payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000}
                async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=10) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
        except aiohttp.ClientError as e:
            logger.error(f"Попытка {attempt + 1} не удалась: {e}")
            if attempt == retries - 1:
                return "Ошибка связи с API. Попробуй позже, ладно?"
            await asyncio.sleep(1)

# Асинхронный поиск через Google Custom Search API
async def google_search(query: str, num_results: int = 3) -> str:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": query, "num": num_results}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as response:
                response.raise_for_status()
                results = (await response.json()).get("items", [])
                formatted_results = [f"[{item['title']}]({item['link']})\n{item.get('snippet', '')}" for item in results]
                return "\n\n".join(formatted_results) if formatted_results else "Ничего не нашёл, попробуй уточнить запрос."
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка Google Search API: {e}")
        return "Ошибка при поиске. Давай попробуем позже?"

# Инициализация базы данных
async def init_db():
    async with aiosqlite.connect("chatbot_context.db") as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS context (
                user_id TEXT PRIMARY KEY,
                history TEXT,
                language TEXT,
                user_level TEXT
            )
        """)
        await conn.commit()

# Получение контекста и уровня знаний
async def get_context(user_id: str) -> tuple[str, str, str]:
    async with aiosqlite.connect("chatbot_context.db") as conn:
        cursor = await conn.execute("SELECT history, language, user_level FROM context WHERE user_id=?", (user_id,))
        result = await cursor.fetchone()
        return (result[0], result[1], result[2]) if result else ("", "ru", "beginner")

# Сохранение контекста и уровня знаний
async def save_context(user_id: str, history: str, language: str, user_level: str):
    async with aiosqlite.connect("chatbot_context.db") as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO context (user_id, history, language, user_level) VALUES (?, ?, ?, ?)",
            (user_id, history[:2000], language, user_level)
        )
        await conn.commit()

# Проверка, адресовано ли сообщение боту
def is_addressed_to_bot(update: Update) -> bool:
    message = update.message
    if not message or not message.text:
        return False
    text = message.text.lower()
    bot_name_pattern = r"^(miki|мики)[,!?.]?\s+"
    has_bot_name = bool(re.match(bot_name_pattern, text))
    is_reply_to_bot = message.reply_to_message and message.reply_to_message.from_user.id == update.effective_user.bot.id
    is_ask_command = text.startswith("/ask")
    return has_bot_name or is_reply_to_bot or is_ask_command

# Обработка вопроса пользователя
async def process_question(question: str, user_id: str, language: str, history: str, user_level: str) -> str:
    if not is_safe_input(question):
        return "Кажется, в запросе что-то подозрительное. Попробуй спросить по-другому!"
    
    search_keywords = ["поиск", "найди", "search", "find"]
    is_search_query = any(keyword in question.lower() for keyword in search_keywords)
    
    if is_search_query:
        search_query = question.lower().replace("поиск", "").replace("найди", "").replace("search", "").replace("find", "").strip()
        response = await google_search(search_query)
        return f"**Вот что я нашёл**:\n{response}"
    else:
        prompt = PROMPT_TEMPLATE.format(user_input=question, context=history, language=language, user_level=user_level)
        response = await query_openrouter(prompt)
        return response if "```" in response else f"**Ответ**:\n{response}"

# Отправка длинных сообщений (лимит Telegram — 4096 символов)
async def send_long_message(update: Update, text: str, parse_mode: str = "Markdown"):
    max_length = 4096
    parts = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    for part in parts:
        await update.message.reply_text(part, parse_mode=parse_mode, disable_web_page_preview=False)

# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Я Miki, твой помощник по программированию и ИТ. Я объясняю всё просто, как другу. "
        "Обращайся ко мне по имени (Miki, Мики), пиши /ask <вопрос> или отвечай на мои сообщения. "
        "Примеры: 'Мики, что такое Python?' или '/ask Как установить Docker?'. "
        "Твой уровень знаний сейчас — новичок, но ты можешь изменить его через /setlevel (beginner, advanced, expert)."
    )

# Обработчик команды /setlang для смены языка
async def set_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Напиши язык, например: /setlang ru или /setlang en")
        return
    language = context.args[0].lower()
    supported_languages = ["ru", "en"]
    if language not in supported_languages:
        await update.message.reply_text(f"Я понимаю только эти языки: {', '.join(supported_languages)}")
        return
    user_id = str(update.effective_user.id)
    history, _, user_level = await get_context(user_id)
    await save_context(user_id, history, language, user_level)
    await update.message.reply_text(f"Язык теперь: {language}")

# Обработчик команды /setlevel для установки уровня знаний
async def set_level(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Напиши уровень знаний, например: /setlevel beginner, /setlevel advanced или /setlevel expert")
        return
    level = context.args[0].lower()
    supported_levels = ["beginner", "advanced", "expert"]
    if level not in supported_levels:
        await update.message.reply_text(f"Я понимаю только эти уровни: {', '.join(supported_levels)}")
        return
    user_id = str(update.effective_user.id)
    history, language, _ = await get_context(user_id)
    await save_context(user_id, history, language, level)
    await update.message.reply_text(f"Твой уровень знаний теперь: {level}")

# Обработчик команды /ask и текстовых сообщений
async def handle_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_addressed_to_bot(update):
        return

    question = (
        " ".join(context.args) if context.args 
        else update.message.text.replace("/ask", "").strip()
    )

    # Удаление имени бота из текста вопроса
    for name in BOT_NAMES:
        question = re.sub(rf"\b{name}\b[,!?.]?\s*", "", question, flags=re.IGNORECASE).strip()

    if not question:
        await update.message.reply_text("Спроси что-нибудь, например, 'Мики, как написать программу?' или используй /ask.")
        return

    user_id = str(update.effective_user.id)
    history, language, user_level = await get_context(user_id)
    response = await process_question(question, user_id, language, history, user_level)

    await save_context(user_id, f"{history}\nUser: {question}\nBot: {response}"[:2000], language, user_level)
    try:
        await send_long_message(update, response)
    except Exception as e:
        logger.error(f"Ошибка отправки сообщения: {e}")
        await update.message.reply_text("Что-то пошло не так с ответом. Попробуй спросить проще!")

# Асинхронная функция для запуска приложения
async def run_application():
    # Проверка наличия ключей
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN не задан в файле .env. Получи его через @BotFather!")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY не задан в файле .env. Получи его на https://openrouter.ai/!")
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        raise ValueError("GOOGLE_API_KEY или GOOGLE_CSE_ID не заданы в файле .env. Получи их на https://console.developers.google.com/ и https://cse.google.com/cse/all!")

    # Инициализация базы данных
    await init_db()

    # Создание приложения
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("setlang", set_language))
    application.add_handler(CommandHandler("setlevel", set_level))
    application.add_handler(CommandHandler("ask", handle_ask))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_ask))
    logger.info("Бот Miki запущен")

    # Инициализация приложения
    await application.initialize()
    # Запуск polling
    await application.start()
    await application.updater.start_polling(allowed_updates=Update.ALL_TYPES)

    # Держим приложение запущенным
    try:
        while True:
            await asyncio.sleep(3600)  # Спим час, чтобы не нагружать CPU
    except KeyboardInterrupt:
        logger.info("Остановка бота...")
        await application.updater.stop()
        await application.stop()
        await application.shutdown()

# Основная функция
def main():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Если цикл уже запущен (как на Render), создаём задачу
            loop.create_task(run_application())
        else:
            # Если цикл не запущен, используем asyncio.run
            asyncio.run(run_application())
    except RuntimeError as e:
        logger.error(f"Ошибка цикла событий: {e}")
        # Если цикл уже запущен, создаём задачу
        asyncio.get_event_loop().create_task(run_application())

if __name__ == "__main__":
    main()
