import datetime as dt, os

import streamlit as st
from cryptography.fernet import InvalidToken

from brain import connectors, llm
from brain.search import load_embedder, search
from brain.store import Vault

st.set_page_config("Другий мозок", "🧠", layout="wide")
VAULT_PATH = os.environ.get("BRAIN_VAULT", "brain.vault")


@st.cache_resource(show_spinner="Завантажую локальну модель пошуку…")
def get_embedder():
    return load_embedder()


emb = get_embedder()
st.title("🧠 Другий мозок")

# ---------- Розблокування сховища ----------
if "vault" not in st.session_state:
    new = not os.path.exists(VAULT_PATH)
    st.info("Створіть пароль для нового зашифрованого сховища." if new else "Введіть пароль до сховища.")
    pw = st.text_input("Пароль", type="password")
    if st.button("Розблокувати", type="primary") and pw:
        try:
            st.session_state.vault = Vault(VAULT_PATH, pw)
            st.session_state.messages = []
            st.rerun()
        except InvalidToken:
            st.error("Невірний пароль.")
    st.caption("Усі дані зберігаються локально у файлі, зашифрованому ключем із вашого пароля (scrypt + Fernet).")
    st.stop()

vault: Vault = st.session_state.vault

# ---------- Бокова панель ----------
with st.sidebar:
    st.header("Джерела")
    st.write(vault.stats() or "Архів порожній")
    if emb is None:
        st.warning("sentence-transformers не встановлено — працює лише пошук за словами.")

    with st.expander("📝 Нотатки (папка .md/.txt)"):
        folder = st.text_input("Шлях до папки", "~/Notes")
        if st.button("Синхронізувати нотатки"):
            st.success(f"Нових/змінених: {connectors.sync_notes(vault, emb, folder)}")

    with st.expander("✉️ Пошта (IMAP, лише читання)"):
        host = st.text_input("IMAP-сервер", "imap.gmail.com")
        user = st.text_input("Логін")
        pw = st.text_input("Пароль застосунку", type="password", help="Пароль не зберігається на диску")
        days = st.slider("За останні, днів", 30, 1095, 365)
        if st.button("Синхронізувати пошту") and user and pw:
            bar = st.progress(0.0)
            try:
                n = connectors.sync_imap(vault, emb, host, user, pw, days=days, progress=bar.progress)
                st.success(f"Нових листів: {n}")
            except Exception as e:
                st.error(f"Помилка IMAP: {e}")

    with st.expander("📅 Календар (.ics)"):
        up = st.file_uploader("Файл .ics", type="ics")
        url = st.text_input("або секретне посилання iCal")
        if st.button("Імпортувати календар"):
            data = up.read() if up else connectors.fetch_ics_url(url) if url else None
            if data:
                st.success(f"Нових подій: {connectors.sync_ics(vault, emb, data)}")

    st.header("Модель")
    provider = st.radio("Де працює LLM", ["ollama", "anthropic"],
                        format_func=lambda p: "Локально (Ollama)" if p == "ollama" else "Хмара (Claude)")
    model = st.text_input("Модель", "llama3.1" if provider == "ollama" else "claude-sonnet-5-5")
    redact = st.checkbox("Маскувати email, телефони, IBAN, картки", True, disabled=provider == "ollama",
                         help="У хмару йдуть лише знайдені фрагменти, а чутливі дані замінюються токенами.")
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Задайте змінну ANTHROPIC_API_KEY.")
    kinds = st.multiselect("Шукати в", ["mail", "calendar", "note"], default=["mail", "calendar", "note"])

    st.header("Небезпечна зона")
    if st.button("🗑 Видалити все сховище"):
        vault.wipe(); st.session_state.clear(); st.rerun()
    if st.button("🔒 Заблокувати"):
        st.session_state.clear(); st.rerun()

# ---------- Чат ----------
def show_sources(hits):
    with st.expander(f"Джерела ({len(hits)})"):
        for h in hits:
            d = dt.datetime.fromtimestamp(h["ts"]).strftime("%Y-%m-%d")
            st.markdown(f"**[{h['n']}] {h['source']} · {h['title']} · {d}**")
            st.text(h["text"][:700])


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("hits"):
            show_sources(m["hits"])

if q := st.chat_input("Що ми домовлялися з постачальником у березні?"):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Шукаю й думаю…"):
            hits, rng = search(vault, q, emb, sources=kinds)
            try:
                text = llm.answer(q, hits, provider, model, redact)
            except Exception as e:
                text = f"Помилка моделі: {e}"
        if rng:
            f = lambda t: dt.datetime.fromtimestamp(t).strftime("%d.%m.%Y")
            st.caption(f"Фільтр за датою з питання: {f(rng[0])} – {f(rng[1] - 86400)}")
        st.markdown(text)
        show_sources(hits)
    st.session_state.messages.append({"role": "assistant", "content": text, "hits": hits})
