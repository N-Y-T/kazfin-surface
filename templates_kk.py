"""Parallel financial-reasoning templates in EN / RU / KK, after FinChain.

Adapted from the FinChain template design (github.com/mbzuai-nlp/finchain,
Apache-2.0). Each template returns the same numbers and the same gold answer
in every language register, so any accuracy difference between registers is
caused by the surface form of the question, not by the task.

Registers produced per instance:
  en      - English (FinChain style)
  ru      - Russian
  kk      - standard written Kazakh
  kk_kb   - kk typed on a Russian keyboard      (derived automatically)
  kk_cs   - Kazakh-Russian code-switched, as people write in chats
            (authored by hand - MUST be reviewed by a native speaker)
  kk_cs_kb- kk_cs typed on a Russian keyboard   (derived automatically)

DRAFT WORDING: the Kazakh and code-switched sentences are first drafts and
need a native-speaker pass before any results are reported.
"""
from __future__ import annotations

import random

from kk_surface import keyboard_flatten

COMPANIES = ["Алатау Құрылыс", "Есіл Агро", "Сарыарқа Логистик",
             "Жетісу Трейд", "Ертіс Энерго"]
COMPANIES_EN = {"Алатау Құрылыс": "Alatau Qurylys", "Есіл Агро": "Esil Agro",
                "Сарыарқа Логистик": "Saryarqa Logistic", "Жетісу Трейд": "Zhetisu Trade",
                "Ертіс Энерго": "Ertis Energo"}


# ---------- number formatting -------------------------------------------- #
def money(x: float, lang: str) -> str:
    """1250000.5 -> '1,250,000.50' (en) or '1 250 000,50' (ru/kk)."""
    s = f"{x:,.2f}"
    if lang == "en":
        return s
    return s.replace(",", " ").replace(".", ",")


def pct(x: float, lang: str) -> str:
    s = f"{x:.2f}"
    return s if lang == "en" else s.replace(".", ",")


def _finish(item: dict) -> dict:
    q = item["question"]
    q["kk_kb"] = keyboard_flatten(q["kk"])
    q["kk_cs_kb"] = keyboard_flatten(q["kk_cs"])
    return item


# ---------- T1: simple interest, single repayment ------------------------ #
def t_simple_interest(rng: random.Random) -> dict:
    co = rng.choice(COMPANIES)
    p = float(rng.randint(5, 200) * 1_000_000)          # tenge
    r = round(rng.uniform(8.0, 24.0), 2)                  # % per year
    t = rng.randint(2, 7)                                 # years
    total = p * (1 + r / 100 * t)
    q = {
        "en": (f"{COMPANIES_EN[co]} LLP takes a loan of KZT {money(p,'en')} at a simple annual "
               f"interest rate of {pct(r,'en')}% for {t} years. Calculate the total "
               f"amount repaid at the end of the term."),
        "ru": (f"ТОО «{co}» получило кредит в размере {money(p,'ru')} тенге на {t} лет "
               f"под {pct(r,'ru')}% годовых (простые проценты). Рассчитайте общую сумму "
               f"к возврату в конце срока."),
        "kk": (f"«{co}» ЖШС {money(p,'kk')} теңге көлемінде {t} жылға жылдық "
               f"{pct(r,'kk')}% жай пайыз мөлшерлемесімен несие алды. Мерзім соңында "
               f"қайтарылатын жалпы соманы есептеңіз."),
        "kk_cs": (f"«{co}» ТОО {money(p,'kk')} теңге кредит алды, {t} жылға, ставкасы "
                  f"жылына {pct(r,'kk')}%, простой процентпен. Срок соңында барлығы "
                  f"қанша қайтарады, посчитайте."),
    }
    steps = [f"r = {r}/100 = {r/100:.4f}",
             f"Total = P × (1 + r × t) = {p:.2f} × (1 + {r/100:.4f} × {t}) = {total:.2f}"]
    return _finish({"template": "simple_interest", "level": "basic",
                    "question": q, "steps": steps, "answer": round(total, 2)})


# ---------- T2: annuity loan, monthly payment ---------------------------- #
def t_annuity_payment(rng: random.Random) -> dict:
    co = rng.choice(COMPANIES)
    p = float(rng.randint(10, 300) * 1_000_000)
    r = round(rng.uniform(10.0, 26.0), 2)
    years = rng.randint(1, 5)
    n = years * 12
    i = r / 100 / 12
    pay = p * i / (1 - (1 + i) ** (-n))
    q = {
        "en": (f"{COMPANIES_EN[co]} LLP takes an annuity loan of KZT {money(p,'en')} for {years} years "
               f"at {pct(r,'en')}% per year, repaid in equal monthly payments. Calculate "
               f"the monthly payment."),
        "ru": (f"ТОО «{co}» взяло аннуитетный кредит на {money(p,'ru')} тенге сроком "
               f"{years} лет под {pct(r,'ru')}% годовых с ежемесячными равными платежами. "
               f"Рассчитайте сумму ежемесячного платежа."),
        "kk": (f"«{co}» ЖШС {money(p,'kk')} теңгеге {years} жылға жылдық {pct(r,'kk')}% "
               f"мөлшерлемемен аннуитеттік несие алды. Төлемдер ай сайын тең сомамен "
               f"жүргізіледі. Ай сайынғы төлем сомасын есептеңіз."),
        "kk_cs": (f"«{co}» ТОО {money(p,'kk')} теңгеге {years} жылға аннуитетный кредит "
                  f"алды, ставка жылына {pct(r,'kk')}%. Платеж ай сайын бірдей. "
                  f"Ежемесячный платеж қанша болады?"),
    }
    steps = [f"i = {r}/100/12 = {i:.6f}", f"n = {years} × 12 = {n}",
             f"A = P × i / (1 − (1 + i)^−n) = {pay:.2f}"]
    return _finish({"template": "annuity_payment", "level": "intermediate",
                    "question": q, "steps": steps, "answer": round(pay, 2)})


# ---------- T3: current liquidity ratio from balance-sheet items --------- #
def t_current_ratio(rng: random.Random) -> dict:
    co = rng.choice(COMPANIES)
    cash = float(rng.randint(20, 400) * 1_000_000)
    ar = float(rng.randint(50, 900) * 1_000_000)
    inv = float(rng.randint(30, 700) * 1_000_000)
    cl = float(rng.randint(80, 1200) * 1_000_000)
    ca = cash + ar + inv
    ratio = ca / cl
    q = {
        "en": (f"Year-end data of {COMPANIES_EN[co]} LLP: cash KZT {money(cash,'en')}, short-term "
               f"receivables KZT {money(ar,'en')}, inventories KZT {money(inv,'en')}, "
               f"short-term liabilities KZT {money(cl,'en')}. Calculate the current "
               f"ratio to two decimal places."),
        "ru": (f"Данные ТОО «{co}» на конец года: денежные средства — {money(cash,'ru')} "
               f"тенге, краткосрочная дебиторская задолженность — {money(ar,'ru')} тенге, "
               f"запасы — {money(inv,'ru')} тенге, краткосрочные обязательства — "
               f"{money(cl,'ru')} тенге. Рассчитайте коэффициент текущей ликвидности "
               f"с точностью до двух знаков."),
        "kk": (f"«{co}» ЖШС-нің жыл соңындағы деректері: ақша қаражаты — "
               f"{money(cash,'kk')} теңге, қысқа мерзімді дебиторлық берешек — "
               f"{money(ar,'kk')} теңге, қорлар — {money(inv,'kk')} теңге, қысқа "
               f"мерзімді міндеттемелер — {money(cl,'kk')} теңге. Ағымдағы өтімділік "
               f"коэффициентін екі ондық белгіге дейін есептеңіз."),
        "kk_cs": (f"«{co}» ТОО-ның год соңындағы данныйлары: деньги {money(cash,'kk')} "
                  f"теңге, дебиторка {money(ar,'kk')} теңге, запастар {money(inv,'kk')} "
                  f"теңге, краткосрочные обязательства {money(cl,'kk')} теңге. Текущая "
                  f"ликвидность коэффициентін есептеңізші, два знака после запятой."),
    }
    steps = [f"Current assets = {cash:.0f} + {ar:.0f} + {inv:.0f} = {ca:.0f}",
             f"Current ratio = {ca:.0f} / {cl:.0f} = {ratio:.4f} ≈ {ratio:.2f}"]
    return _finish({"template": "current_ratio", "level": "basic",
                    "question": q, "steps": steps, "answer": round(ratio, 2)})


PEOPLE = ["Айгерім", "Ерлан", "Дана", "Нұрлан", "Мадина"]
PEOPLE_EN = {"Айгерім": "Aigerim", "Ерлан": "Erlan", "Дана": "Dana",
             "Нұрлан": "Nurlan", "Мадина": "Madina"}
MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July",
             "August", "September", "October", "November", "December"]


def date_str(d, lang: str) -> str:
    """15 March 2026 (en) or 15.03.2026 (ru/kk)."""
    return f"{d.day} {MONTHS_EN[d.month - 1]} {d.year}" if lang == "en" else d.strftime("%d.%m.%Y")


def _annuity(p: float, r: float, n: int) -> float:
    i = r / 100 / 12
    return p * i / (1 - (1 + i) ** (-n))


# ---------- T4: bank deposit with monthly capitalisation ----------------- #
def t_deposit_compound(rng: random.Random) -> dict:
    who = rng.choice(PEOPLE)
    p = float(rng.randint(1, 50) * 100_000)
    r = round(rng.uniform(10.0, 17.0), 2)
    m = rng.choice([6, 9, 12, 18, 24, 36])
    fv = p * (1 + r / 100 / 12) ** m
    q = {
        "en": (f"{PEOPLE_EN[who]} places KZT {money(p,'en')} on a bank deposit at {pct(r,'en')}% "
               f"per year with monthly capitalisation of interest for {m} months. Calculate "
               f"the deposit balance at the end of the term."),
        "ru": (f"{who} вносит на банковский депозит {money(p,'ru')} тенге под {pct(r,'ru')}% "
               f"годовых с ежемесячной капитализацией процентов на {m} месяцев. Рассчитайте "
               f"сумму на депозите в конце срока."),
        "kk": (f"{who} банк салымына {money(p,'kk')} теңге салды: мөлшерлеме жылдық "
               f"{pct(r,'kk')}%, пайыздар ай сайын капиталданады, мерзімі {m} ай. Мерзім "
               f"соңындағы салым сомасын есептеңіз."),
        "kk_cs": (f"{who} депозитке {money(p,'kk')} теңге салды, годовой {pct(r,'kk')}%, "
                  f"капитализация ай сайын, срок {m} ай. Срок біткенде депозитте қанша "
                  f"болады?"),
    }
    steps = [f"i = {r}/100/12 = {r/1200:.6f}",
             f"FV = P × (1 + i)^m = {p:.2f} × (1 + {r/1200:.6f})^{m} = {fv:.2f}"]
    return _finish({"template": "deposit_compound", "level": "basic",
                    "question": q, "steps": steps, "answer": round(fv, 2)})


# ---------- T5: total interest (overpayment) on an annuity loan ---------- #
def t_loan_overpayment(rng: random.Random) -> dict:
    co = rng.choice(COMPANIES)
    p = float(rng.randint(10, 300) * 1_000_000)
    r = round(rng.uniform(10.0, 26.0), 2)
    years = rng.randint(1, 7)
    n = years * 12
    a = _annuity(p, r, n)
    over = a * n - p
    q = {
        "en": (f"{COMPANIES_EN[co]} LLP takes an annuity loan of KZT {money(p,'en')} for {years} "
               f"years at {pct(r,'en')}% per year, repaid in equal monthly payments. Calculate "
               f"the total interest paid over the life of the loan."),
        "ru": (f"ТОО «{co}» взяло аннуитетный кредит на {money(p,'ru')} тенге сроком {years} "
               f"лет под {pct(r,'ru')}% годовых с ежемесячными равными платежами. Рассчитайте "
               f"переплату по кредиту, то есть общую сумму процентов за весь срок."),
        "kk": (f"«{co}» ЖШС {money(p,'kk')} теңгеге {years} жылға жылдық {pct(r,'kk')}% "
               f"мөлшерлемемен аннуитеттік несие алды, төлемдер ай сайын тең сомамен "
               f"жүргізіледі. Несиенің бүкіл мерзімінде төленетін пайыздардың жалпы сомасын "
               f"(артық төлемді) есептеңіз."),
        "kk_cs": (f"«{co}» ТОО {money(p,'kk')} теңгеге {years} жылға аннуитетный кредит алды, "
                  f"ставка жылына {pct(r,'kk')}%, платеж ай сайын бірдей. Весь срок бойы "
                  f"переплата қанша болады, яғни процентке барлығы қанша төлейді?"),
    }
    steps = [f"A = P × i / (1 − (1 + i)^−n) = {a:.2f}, n = {n}",
             f"Interest = A × n − P = {a:.2f} × {n} − {p:.2f} = {over:.2f}"]
    return _finish({"template": "loan_overpayment", "level": "intermediate",
                    "question": q, "steps": steps, "answer": round(over, 2)})


# ---------- T6: outstanding principal after k annuity payments ----------- #
def t_loan_balance(rng: random.Random) -> dict:
    co = rng.choice(COMPANIES)
    p = float(rng.randint(10, 300) * 1_000_000)
    r = round(rng.uniform(10.0, 26.0), 2)
    years = rng.randint(2, 7)
    n = years * 12
    k = rng.randint(3, n - 3)
    i = r / 100 / 12
    a = _annuity(p, r, n)
    bal = p * (1 + i) ** k - a * ((1 + i) ** k - 1) / i
    q = {
        "en": (f"{COMPANIES_EN[co]} LLP takes an annuity loan of KZT {money(p,'en')} for {years} "
               f"years at {pct(r,'en')}% per year, repaid in equal monthly payments. Calculate "
               f"the outstanding principal after {k} monthly payments have been made."),
        "ru": (f"ТОО «{co}» взяло аннуитетный кредит на {money(p,'ru')} тенге сроком {years} "
               f"лет под {pct(r,'ru')}% годовых с ежемесячными равными платежами. Рассчитайте "
               f"остаток основного долга после внесения {k} ежемесячных платежей."),
        "kk": (f"«{co}» ЖШС {money(p,'kk')} теңгеге {years} жылға жылдық {pct(r,'kk')}% "
               f"мөлшерлемемен аннуитеттік несие алды, төлемдер ай сайын тең сомамен "
               f"жүргізіледі. {k} ай сайынғы төлем жасалғаннан кейінгі негізгі борыштың "
               f"қалдығын есептеңіз."),
        "kk_cs": (f"«{co}» ТОО {money(p,'kk')} теңгеге {years} жылға аннуитетный кредит алды, "
                  f"ставка жылына {pct(r,'kk')}%, платеж ай сайын бірдей. {k} платеж "
                  f"төленгеннен кейін остаток основного долга қанша қалады?"),
    }
    steps = [f"i = {i:.6f}, A = {a:.2f}",
             f"B_k = P(1+i)^k − A((1+i)^k − 1)/i, k = {k} → {bal:.2f}"]
    return _finish({"template": "loan_balance", "level": "advanced",
                    "question": q, "steps": steps, "answer": round(bal, 2)})


# ---------- T7: contractual penalty for late payment, with a cap -------- #
def t_contract_penalty(rng: random.Random) -> dict:
    import datetime as dt
    co = rng.choice(COMPANIES)
    debt = float(rng.randint(2, 200) * 500_000)
    rate = rng.choice([0.05, 0.1, 0.15, 0.2, 0.3])        # % per day
    cap = rng.choice([10, 15, 20])                         # % of debt
    due = dt.date(2025, 1, 1) + dt.timedelta(days=rng.randint(0, 540))
    days = rng.randint(10, 200)
    paid = due + dt.timedelta(days=days)
    raw = debt * rate / 100 * days
    pen = min(raw, debt * cap / 100)
    q = {
        "en": (f"Under a supply contract, {COMPANIES_EN[co]} LLP had to pay KZT {money(debt,'en')}; "
               f"the payment deadline was {date_str(due,'en')}. Payment was received on "
               f"{date_str(paid,'en')}. The contract sets a penalty of {pct(rate,'en')}% of the "
               f"unpaid amount for each day of delay, counted from the day after the deadline up to "
               f"and including the payment date, but not more than {cap}% of the unpaid amount. "
               f"Calculate the penalty."),
        "ru": (f"По договору поставки ТОО «{co}» должно было оплатить {money(debt,'ru')} тенге, "
               f"срок оплаты — {date_str(due,'ru')}. Оплата поступила {date_str(paid,'ru')}. "
               f"Договором предусмотрена неустойка {pct(rate,'ru')}% от неоплаченной суммы за "
               f"каждый день просрочки, начиная со дня, следующего за сроком оплаты, по день "
               f"оплаты включительно, но не более {cap}% от неоплаченной суммы. Рассчитайте "
               f"сумму неустойки."),
        "kk": (f"Жеткізу шарты бойынша «{co}» ЖШС {money(debt,'kk')} теңге төлеуге тиіс еді, "
               f"төлем мерзімі — {date_str(due,'kk')}. Төлем {date_str(paid,'kk')} күні түсті. "
               f"Шартта мерзімі өткен әрбір күн үшін төленбеген соманың {pct(rate,'kk')}% "
               f"мөлшерінде тұрақсыздық айыбы көзделген: ол төлем мерзімінен кейінгі күннен "
               f"бастап төлем күнін қоса алғанда есептеледі, бірақ төленбеген соманың "
               f"{cap}%-ынан аспайды. Тұрақсыздық айыбының сомасын есептеңіз."),
        "kk_cs": (f"Поставка договоры бойынша «{co}» ТОО {money(debt,'kk')} теңге төлеу керек еді, "
                  f"срок оплаты — {date_str(due,'kk')}. Оплата {date_str(paid,'kk')} күні келді. "
                  f"Договорда неустойка бар: просрочканың әр күніне неоплаченная сумманың "
                  f"{pct(rate,'kk')}%-ы, срок біткен келесі күннен бастап оплата күнін қоса, "
                  f"бірақ сумманың {cap}%-ынан көп емес. Неустойка қанша болады?"),
    }
    steps = [f"Days of delay = {date_str(paid,'en')} − {date_str(due,'en')} = {days}",
             f"Penalty before cap = {debt:.2f} × {rate}% × {days} = {raw:.2f}",
             f"Cap = {cap}% × {debt:.2f} = {debt*cap/100:.2f}; penalty = {pen:.2f}"]
    return _finish({"template": "contract_penalty", "level": "intermediate",
                    "question": q, "steps": steps, "answer": round(pen, 2)})


# ---------- T8: VAT included in a gross amount -------------------------- #
def t_vat_extraction(rng: random.Random) -> dict:
    co = rng.choice(COMPANIES)
    v = rng.choice([12, 16])
    gross = float(rng.randint(100, 50_000) * 1_000)
    vat = gross * v / (100 + v)
    q = {
        "en": (f"An invoice issued by {COMPANIES_EN[co]} LLP totals KZT {money(gross,'en')}, "
               f"including VAT at {v}%. Calculate the amount of VAT included in the invoice."),
        "ru": (f"Сумма счёта-фактуры ТОО «{co}» составляет {money(gross,'ru')} тенге, включая "
               f"НДС по ставке {v}%. Рассчитайте сумму НДС в составе счёта."),
        "kk": (f"«{co}» ЖШС шот-фактурасының сомасы {v}% мөлшерлемесі бойынша ҚҚС-ты қоса "
               f"алғанда {money(gross,'kk')} теңге. Шоттың құрамындағы ҚҚС сомасын есептеңіз."),
        "kk_cs": (f"«{co}» ТОО счет-фактурасы {money(gross,'kk')} теңге, НДС {v}% ішінде. "
                  f"Осы сумманың ішіндегі НДС қанша, выделить етіп беріңізші."),
    }
    steps = [f"VAT = G × v / (100 + v) = {gross:.2f} × {v}/{100+v} = {vat:.2f}"]
    return _finish({"template": "vat_extraction", "level": "basic",
                    "question": q, "steps": steps, "answer": round(vat, 2)})


# ---------- T9: straight-line depreciation, carrying amount ------------- #
def t_depreciation(rng: random.Random) -> dict:
    co = rng.choice(COMPANIES)
    cost = float(rng.randint(5, 500) * 1_000_000)
    life = rng.randint(4, 15)
    salv = float(round(cost * rng.uniform(0.0, 0.15) / 100_000) * 100_000)
    k = rng.randint(1, life - 1)
    carry = cost - (cost - salv) / life * k
    q = {
        "en": (f"{COMPANIES_EN[co]} LLP bought equipment for KZT {money(cost,'en')}. Its useful life "
               f"is {life} years and its residual value is KZT {money(salv,'en')}. Using the "
               f"straight-line method, calculate the carrying amount after {k} full years."),
        "ru": (f"ТОО «{co}» приобрело оборудование за {money(cost,'ru')} тенге. Срок полезного "
               f"использования — {life} лет, ликвидационная стоимость — {money(salv,'ru')} тенге. "
               f"Рассчитайте балансовую стоимость оборудования через {k} полных лет при "
               f"прямолинейном методе амортизации."),
        "kk": (f"«{co}» ЖШС жабдықты {money(cost,'kk')} теңгеге сатып алды. Пайдалы қызмет "
               f"мерзімі — {life} жыл, тарату құны — {money(salv,'kk')} теңге. Біркелкі (тура "
               f"сызықты) амортизация әдісімен {k} толық жылдан кейінгі жабдықтың баланстық "
               f"құнын есептеңіз."),
        "kk_cs": (f"«{co}» ТОО оборудование алды {money(cost,'kk')} теңгеге. Срок полезного "
                  f"использования {life} жыл, ликвидационная стоимость {money(salv,'kk')} теңге. "
                  f"Прямолинейный методпен амортизация есептесек, {k} жылдан кейін балансовая "
                  f"стоимость қанша болады?"),
    }
    steps = [f"Annual depreciation = ({cost:.0f} − {salv:.0f}) / {life} = {(cost-salv)/life:.2f}",
             f"Carrying amount = {cost:.0f} − {(cost-salv)/life:.2f} × {k} = {carry:.2f}"]
    return _finish({"template": "depreciation", "level": "basic",
                    "question": q, "steps": steps, "answer": round(carry, 2)})


# ---------- T10: return on average equity ------------------------------ #
def t_roe(rng: random.Random) -> dict:
    co = rng.choice(COMPANIES)
    e0 = float(rng.randint(100, 5000) * 1_000_000)
    e1 = float(round(e0 * rng.uniform(0.85, 1.35) / 1_000_000) * 1_000_000)
    ni = float(round((e0 + e1) / 2 * rng.uniform(0.02, 0.30) / 100_000) * 100_000)
    roe = ni / ((e0 + e1) / 2) * 100
    q = {
        "en": (f"{COMPANIES_EN[co]} LLP reported net profit of KZT {money(ni,'en')} for the year. "
               f"Equity was KZT {money(e0,'en')} at the beginning of the year and KZT "
               f"{money(e1,'en')} at the end. Calculate return on equity (ROE) based on average "
               f"equity, in percent."),
        "ru": (f"Чистая прибыль ТОО «{co}» за год составила {money(ni,'ru')} тенге. Собственный "
               f"капитал на начало года — {money(e0,'ru')} тенге, на конец года — "
               f"{money(e1,'ru')} тенге. Рассчитайте рентабельность собственного капитала (ROE) "
               f"по среднему значению капитала, в процентах."),
        "kk": (f"«{co}» ЖШС-нің жылдық таза пайдасы {money(ni,'kk')} теңге болды. Меншікті "
               f"капитал жыл басында — {money(e0,'kk')} теңге, жыл соңында — {money(e1,'kk')} "
               f"теңге. Капиталдың орташа шамасы бойынша меншікті капиталдың рентабельділігін "
               f"(ROE) пайызбен есептеңіз."),
        "kk_cs": (f"«{co}» ТОО-ның чистая прибылі жылына {money(ni,'kk')} теңге. Собственный "
                  f"капитал год басында {money(e0,'kk')} теңге, год соңында {money(e1,'kk')} "
                  f"теңге. Средний капиталмен ROE қанша процент шығады?"),
    }
    steps = [f"Average equity = ({e0:.0f} + {e1:.0f}) / 2 = {(e0+e1)/2:.2f}",
             f"ROE = {ni:.0f} / {(e0+e1)/2:.2f} × 100 = {roe:.4f}%"]
    return _finish({"template": "roe", "level": "intermediate",
                    "question": q, "steps": steps, "answer": round(roe, 2)})


# ---------- T11: net present value of a project ------------------------- #
def t_npv(rng: random.Random) -> dict:
    co = rng.choice(COMPANIES)
    while True:
        inv = float(rng.randint(50, 2000) * 1_000_000)
        n = rng.randint(3, 5)
        cfs = [float(round(inv * rng.uniform(0.15, 0.55) / 1_000_000) * 1_000_000) for _ in range(n)]
        d = rng.choice([8, 10, 12, 14, 15, 16, 18, 20])
        npv = -inv + sum(cf / (1 + d / 100) ** (t + 1) for t, cf in enumerate(cfs))
        if abs(npv) >= 0.05 * inv:          # keep the answer away from zero
            break

    def flows(lang: str) -> str:
        if lang == "en":
            return "; ".join(f"year {t+1} — KZT {money(cf,'en')}" for t, cf in enumerate(cfs))
        if lang == "ru":
            return "; ".join(f"{t+1}-й год — {money(cf,'ru')} тенге" for t, cf in enumerate(cfs))
        return "; ".join(f"{t+1}-жыл — {money(cf,'kk')} теңге" for t, cf in enumerate(cfs))

    q = {
        "en": (f"{COMPANIES_EN[co]} LLP is considering a project with an initial investment of KZT "
               f"{money(inv,'en')}. Expected net cash flows: {flows('en')}. The discount rate is "
               f"{d}%. Calculate the project's net present value (NPV)."),
        "ru": (f"ТОО «{co}» рассматривает проект с первоначальными инвестициями {money(inv,'ru')} "
               f"тенге. Ожидаемые чистые денежные потоки: {flows('ru')}. Ставка дисконтирования — "
               f"{d}%. Рассчитайте чистую приведённую стоимость проекта (NPV)."),
        "kk": (f"«{co}» ЖШС бастапқы инвестициясы {money(inv,'kk')} теңге болатын жобаны "
               f"қарастырып отыр. Күтілетін таза ақша ағындары: {flows('kk')}. Дисконттау "
               f"мөлшерлемесі — {d}%. Жобаның таза келтірілген құнын (NPV) есептеңіз."),
        "kk_cs": (f"«{co}» ТОО бір проектті қарап жатыр, инвестиция {money(inv,'kk')} теңге. "
                  f"Денежный потоктар: {flows('kk')}. Ставка дисконтирования {d}%. NPV қанша "
                  f"шығады?"),
    }
    steps = [f"PV(CF_t) = CF_t / (1 + {d/100})^t",
             f"NPV = −{inv:.0f} + Σ PV = {npv:.2f}"]
    return _finish({"template": "npv", "level": "intermediate",
                    "question": q, "steps": steps, "answer": round(npv, 2)})


# ---------- T12: FX loss on revaluation of a USD loan ------------------- #
def t_fx_revaluation(rng: random.Random) -> dict:
    co = rng.choice(COMPANIES)
    usd = float(rng.randint(10, 2000) * 10_000)
    r0 = round(rng.uniform(450.0, 550.0), 2)
    r1 = round(r0 + rng.uniform(3.0, 40.0), 2)
    loss = usd * (r1 - r0)
    q = {
        "en": (f"{COMPANIES_EN[co]} LLP has a bank loan of USD {money(usd,'en')}. The exchange rate "
               f"was KZT {pct(r0,'en')} per USD at the beginning of the quarter and KZT "
               f"{pct(r1,'en')} per USD at the end. Calculate the foreign exchange loss on "
               f"revaluation of the loan, in tenge."),
        "ru": (f"У ТОО «{co}» есть банковский кредит в размере {money(usd,'ru')} долларов США. "
               f"Курс на начало квартала — {pct(r0,'ru')} тенге за доллар, на конец квартала — "
               f"{pct(r1,'ru')} тенге за доллар. Рассчитайте курсовую разницу (убыток) от "
               f"переоценки кредита в тенге."),
        "kk": (f"«{co}» ЖШС-нің {money(usd,'kk')} АҚШ доллары мөлшеріндегі банк несиесі бар. "
               f"Тоқсан басындағы бағам — 1 доллар үшін {pct(r0,'kk')} теңге, тоқсан "
               f"соңындағы бағам — {pct(r1,'kk')} теңге. Несиені қайта бағалаудан туындайтын "
               f"бағамдық айырманы (шығынды) теңгемен есептеңіз."),
        "kk_cs": (f"«{co}» ТОО-да {money(usd,'kk')} долларлық кредит бар. Квартал басында курс "
                  f"{pct(r0,'kk')} теңге болған, квартал соңында {pct(r1,'kk')} теңге. "
                  f"Переоценкадан курсовая разница қанша, убыток теңгемен?"),
    }
    steps = [f"FX loss = {usd:.2f} × ({r1} − {r0}) = {loss:.2f}"]
    return _finish({"template": "fx_revaluation", "level": "basic",
                    "question": q, "steps": steps, "answer": round(loss, 2)})


TEMPLATES = [t_simple_interest, t_annuity_payment, t_current_ratio, t_deposit_compound,
             t_loan_overpayment, t_loan_balance, t_contract_penalty, t_vat_extraction,
             t_depreciation, t_roe, t_npv, t_fx_revaluation]
REGISTERS = ["en", "ru", "kk", "kk_kb", "kk_cs", "kk_cs_kb"]
