"""
Advanced Prompt Engineering Techniques
=======================================
Demonstrates: Self-Consistency, Generated Knowledge Prompting,
Tree of Thoughts (ToT), and Program-Aided Language Models (PAL).
"""

from collections import Counter
from getpass import getpass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tot.base import ToTChain
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity
from langchain_experimental.pal_chain.base import PALChain


def init_llm():
    course_api_key = getpass(prompt="Введите ваш ключ, полученный в боте курса: ")
    return ChatOpenAI(
        api_key=course_api_key,
        model="gpt-4o-mini",
        base_url="https://api.vsellm.ru/",
    )


# ──────────────────────────────────────────────
# 1. Self-Consistency
# ──────────────────────────────────────────────

def self_consistency(llm, question: str, repetitions: int = 5) -> str:
    """Generate multiple answers and return the most common one."""
    answers = []
    for _ in range(repetitions):
        response = llm.invoke(question)
        answers.append(response.content)
    print("All answers:", answers)
    most_common = Counter(answers).most_common(1)[0][0]
    print(f"Most common answer: {most_common}")
    return most_common


def demo_self_consistency(llm):
    print("\n=== Self-Consistency ===\n")

    # Good example — random hallucinations are filtered out
    print("--- Good example: longest river in Russia ---")
    self_consistency(llm, "Самая длинная река в России. Ответь одним словом")

    # Bad example — model never gets the right answer
    print("\n--- Bad example: age puzzle ---")
    problem = (
        "Когда мне было 6 лет, моя сестра была в два раза младше меня. "
        "Сейчас мне 70 лет. Сколько лет моей сестре?\n"
        "Выведи в ответ только число."
    )
    self_consistency(llm, problem)


# ──────────────────────────────────────────────
# 2. Generated Knowledge Prompting
# ──────────────────────────────────────────────

def split_q(questions: str) -> list[str]:
    return questions.split("\n")


def demo_generated_knowledge(llm):
    print("\n=== Generated Knowledge Prompting ===\n")

    tematic = "Онлайн образование"

    # Baseline — simple prompt
    print("--- Baseline (simple prompt) ---")
    simple_prompt = PromptTemplate.from_template(
        "Ты it блогер. Напиши статью про: {tematic}"
    )
    simple_chain = simple_prompt | llm | StrOutputParser()
    print(simple_chain.invoke({"tematic": tematic}))

    # Step 1: generate questions
    prompt1 = PromptTemplate.from_template(
        "Ты it блогер, журналист. Поставь 3 интересных вопроса, чтобы раскрыть тему: {tematic}\n"
        "Разделяй вопросы переносом на новую строку"
    )
    chain1 = prompt1 | llm | StrOutputParser() | split_q
    questions = chain1.invoke({"tematic": tematic})
    print("\n--- Generated questions ---")
    print(questions)

    # Step 2: expert answers (batch)
    prompt2 = PromptTemplate.from_template(
        "Ты специалист в области {tematic}, занимаешься этим много лет и хочешь ярко и интересно "
        "рассказать о своем деле жизни.\n"
        "Тебе надо ответить на вопрос о твоем любимом деле.\n"
        "Используй весь свой опыт и знания, чтобы рассказать ярко, интересно, достаточно развернуто.\n"
        "Постарайся дать ответ в 10 предложениях.\n"
        "Вопрос: {question}\nОтвет: "
    )
    chain2 = prompt2 | llm | StrOutputParser()
    answers = chain2.batch(
        [{"tematic": tematic, "question": q} for q in questions]
    )

    # Combine Q&A into context
    context = "\n\n".join(
        f"{q}\n{a}" for q, a in zip(questions, answers)
    )

    # Step 3: final blog article
    prompt3 = PromptTemplate.from_template(
        "Ты ведешь популярный блог про {tematic}, занимаешься этим много лет и хочешь ярко и интересно\n"
        "рассказать о своем деле жизни.\n"
        "Тебе дали ответы на вопросы. Преврати их в интересную статью для блога на одну страницу.\n"
        "Придумай к ней кликбэйтный заголовок.\n"
        "Ответы на вопросы: {context}\n"
    )
    chain3 = prompt3 | llm | StrOutputParser()
    article = chain3.invoke({"tematic": tematic, "context": context})
    print("\n--- Generated article ---")
    print(article)


# ──────────────────────────────────────────────
# 3. Tree of Thoughts — Game of 24
# ──────────────────────────────────────────────

class Game24Checker(ToTChecker):
    def evaluate(self, problem_description: str, thoughts: tuple[str, ...] = ()) -> ThoughtValidity:
        last_thought = thoughts[-1]
        nums = last_thought.split(" Remaining numbers: ")[-1].split(" ")

        if len(nums) == 1:
            if float(nums[0]) == 24:
                return ThoughtValidity.VALID_FINAL
            return ThoughtValidity.INVALID

        return ThoughtValidity.VALID_INTERMEDIATE


def demo_tree_of_thoughts(llm):
    print("\n=== Tree of Thoughts — Game of 24 ===\n")

    puzzle = "20 3 2 1"
    problem_description = f"""{puzzle}
Дано выражение puzzle, в котором через пробел написаны 4 числа. Нам нужно на каждом шаге взять два числа
и провести между ними одну арифметическую опреацию: сложение, вычитание, умножение (+, -, *).
Далее результат арифметической операции мы записываем вместо двух чисел, с которыми была проведена операция.
В конечном счете у нас останется одно число.
Мы хотим, чтобы оно было равно 24.
Далее нужно написать результат арифметической операции рядом с другими числами.
Например у нас было 4 числа: 1 1 3 4
мы решили сложить единицу и другую единицу, значит результатом шага будет: 2 3 4. Далее мы отправим эти три числа : 2 3 4 на следующий шаг.
Перемножим 3 и 4 получим 12 2. Отправим на следующий шаг.
Перемножим 12 и 2 и получим число 24.
Мы хотим получить в конечном счете число 24.
Мы можем возвращаться с неудачных шагов назад. Т.е. если на последнем шаге мы получили числа: 10 8 мы не сможем получить 24 применяя
к этим двум арифметические операции, значит мы можем вернуться на предыдущий шаг и попробовать другое направление.
Нельзя выполнять более одной арифметической операции за шаг.
В конце каждого шага выводи получившиеся числа после строчки " Remaining numbers: "
""".strip()

    tot_chain = ToTChain(
        llm=llm,
        checker=Game24Checker(),
        k=50,
        c=3,
        verbose=True,
        verbose_llm=False,
    )
    result = tot_chain.invoke({"problem_description": problem_description})
    print(result)


# ──────────────────────────────────────────────
# 4. PAL — Program-Aided Language Models
# ──────────────────────────────────────────────

def demo_pal(llm):
    print("\n=== PAL — Program-Aided Language Models ===\n")

    question = (
        "У меня есть стул, две картофелины, цветная капуста, качан салата, "
        "два стола, капуста, две луковицы и три холодильника. Сколько у меня овощей?"
    )

    # Baseline
    print("--- Baseline (plain LLM) ---")
    print(llm.invoke(question).content)

    # PAL
    print("\n--- PAL result ---")
    add = (
        "\n\nIMPORTANT: Generate ONLY the Python code. DO NOT include ```python or ``` "
        "markdown fences or any other extra text. Just the raw Python function `solution()`."
    )
    pal_chain = PALChain.from_math_prompt(llm, allow_dangerous_code=True, verbose=True)
    result = pal_chain.invoke(question + add)
    print(result)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    llm = init_llm()

    demo_self_consistency(llm)
    demo_generated_knowledge(llm)
    demo_tree_of_thoughts(llm)
    demo_pal(llm)
