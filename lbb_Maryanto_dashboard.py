import streamlit as st

import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# --- Load Environment ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SECTORS_API_KEY = os.getenv("SECTORS_API_KEY")

# --- Constants ---
BASE_URL = "https://api.sectors.app/v1"
HEADERS = {"Authorization": SECTORS_API_KEY}

# --- Init LLM ---
llm = ChatGroq(
    temperature=0.8,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)


# ===================== UTILS ===================== #
def fetch_data(endpoint: str, params: dict = None):
    """Generic function to fetch data from Sectors API."""
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()


def run_llm(prompt_template: str, data: pd.DataFrame):
    """Format prompt with data and invoke LLM."""
    prompt = PromptTemplate.from_template(prompt_template).format(data=data.to_string(index=False))
    return llm.invoke(prompt).content


def clean_python_code(raw_code: str):
    """Cleans LLM-generated Python code block."""
    return raw_code.strip().strip("```").replace("python", "").strip()


# ===================== SECTIONS ===================== #
def sidebar_selector():

    """Sidebar untuk subsektor & perusahaan."""

    st.sidebar.title("📌 Pilihan Analisis")

    subsectors = fetch_data("subsectors/")
    subsector_list = pd.DataFrame(subsectors)["subsector"].sort_values().tolist()

    ## streamlit UI
    selected_subsector = st.sidebar.selectbox("🔽 Pilih Subsector", subsector_list)

    companies = fetch_data("companies/", params={"sub_sector": selected_subsector})
    companies_df = pd.DataFrame(companies)
    company_options = companies_df["symbol"] + " - " + companies_df["company_name"]

    ## streamlit UI
    selected_company = st.sidebar.selectbox("🏢 Pilih Perusahaan", company_options)

    return selected_company.split(" - ")[0]  # return symbol


def financial_summary(symbol: str):

    """Ringkasan eksekutif keuangan dari LLM."""

    financials = pd.DataFrame(fetch_data(f"financials/quarterly/{symbol}/",
                                         params={"n_quarters": "6",
                                                "report_date": "2024-03-31"}))

    prompt = """

    Anda adalah seorang analis keuangan yang sangat pintar.
    Berdasarkan data keuangan kuartalan berikut (dalam jutaan Rupiah):

    {data}

    Tuliskan ringkasan eksekutif berdasarkan poin dibawah ini.
    Fokus pada:

    1. Analisis Tren Waktu
    - Pertumbuhan : Revenue, Net Premium Income, Operating Expense, Net Cash Flow.
    - Profitabilitas : Laba kotor, laba operasi, laba bersih (kalau tersedia).

    2. Rasio Keuangan Utama
    Profitability :
        - Net Margin = (Revenue - Operating Expense - Provision) / Revenue.
        - ROA = Net Income / Total Assets.
        - ROE = Net Income / Equity (kalau ada equity).

    Liquidity :
        - Current Ratio = Total Current Asset / Total Non-Current Liabilities.
        - Cash Ratio = Cash and Short-Term Investments / Total Current Asset.

    Efficiency :
        - Expense Ratio = Operating Expense / Revenue.
        - Combined Ratio (khusus asuransi) = (Premium Expense + Operating Expense) / Premium Income.

    3. Arus Kas
    - Analisis Operating vs Investing vs Financing Cash Flow : apakah perusahaan menghasilkan kas dari operasi atau bergantung dari utang/pendanaan.
    - Free Cash Flow = Operating Cash Flow - Capital Expenditure.
    - Net Cash Flow Trend : apakah cadangan kas tumbuh atau berkurang.

    4. Health Check
    - Bandingkan antar periode (QoQ, YoY).
    - Cek apakah perusahaan:
        - Sustainable : pendapatan tumbuh, beban terkendali.
        - Likuid : punya kas cukup untuk jangka pendek.
        - Efisien : Opex tidak lebih cepat tumbuh dari pendapatan.
        - Sehat dari sisi cashflow : Operating CF positif secara konsisten.

    5. Forecasting
    - Forecast Revenue & Cash Flow : gunakan regresi atau ARIMA.
    - Scenario analysis : misalnya dampak kenaikan Opex 10% ke net cash flow.

    """
    summary = run_llm(prompt, financials)

    with st.expander("💡 Ringkasan Keuangan"):
        st.markdown(summary)

    return financials


def revenue_trend(symbol: str, financials: pd.DataFrame):

    """Generate line plot untuk tren pendapatan."""

    data_sample = financials[['date', 'revenue', 'operating_expense', 'net_cash_flow']].dropna()

    prompt = f"""

    Anda adalah seorang programmer Python yang ahli dalam visualisasi data.

    Berikut adalah data revenue, operating_expense, dan net_cash_flow perusahaan:

    {data_sample}

    Buat sebuah skrip Python menggunakan matplotlib untuk menghasilkan line plot. 
    Instruksi:
    - Sumbu X adalah 'date' dan di urutkan dari periode awal sampai periode terakhir
    - Sumbu Y adalah 'revenue', 'operating_expense', dan 'net_cash_flow'
    - Dibuat dalam 1 chart dan berikan warna yang berbeda revenue : biru, operating_expense : merah, net_cash_flow : hijau
    - Berikan poin di sumbu y nya 
    - Berikan nilai y dalam miliar rupiah pada masing-masing line
    - Buat semua dalam bentuk line bukan garis putus-putus
    - Untuk sumbu x di rotasi 45 derajat
    - PENTING: Simpan plot ke dalam variabel bernama `fig`. Contoh: `fig, ax = plt.subplots()`
     
    Tulis HANYA kode Python yang bisa langsung dieksekusi. Jangan sertakan penjelasan apapun.

    """
    code = clean_python_code(llm.invoke(prompt).content)

    with st.expander("📊 Visualisasi Trends"):
        exec_locals = {}
        exec(code, {}, exec_locals)
        st.pyplot(exec_locals["fig"])


def trend_analysis(financials: pd.DataFrame):
    """Interpretasi tren keuangan (LLM)."""
    prompt = """

    Anda adalah seorang analis keuangan yang hebat.
    Berdasarkan data kuartalan berikut:

    {data}

    Analisis tren utama yang muncul dari data tersebut. Fokus pada pergerakan revenue, operating_expense, dan net_cash_flow.
    Sajikan analisis dalam 3 poin. Tuliskan dalam bahasa yang singkat, padat, jelas, dan mudah untuk dapat dipahami.

    """
    analysis = run_llm(prompt, financials)
    with st.expander("🔎 Interpretasi Tren Keuangan"):
        st.markdown(analysis)


def risk_analysis(financials: pd.DataFrame):

    """Analisis risiko keuangan (LLM)."""

    prompt = """

    Anda adalah seorang analis risiko keuangan yang skeptis.
    Periksa data keuangan berikut dengan teliti:

    {data}
    
    Identifikasi 1 sampai 3 potensi risiko atau "red flags" yang perlu diwaspadai dari data tersebut. 
    Jelaskan dalam satu kalimat singkat, padat, jelas dan mudah untuk dipahami.

    """
    risks = run_llm(prompt, financials)
    with st.expander("⚠️ Potensi Risiko Keuangan"):
        st.markdown(risks)


# ===================== MAIN APP ===================== #
def main():
    symbol = sidebar_selector()

    if st.sidebar.button("🔍 Lihat Insight"):
        financials = financial_summary(symbol)
        revenue_trend(symbol, financials)
        trend_analysis(financials)
        risk_analysis(financials)


if __name__ == "__main__":
    main()
