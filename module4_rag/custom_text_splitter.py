import re


def get_content_from_sections(text: str) -> list[str]:
    """Parse structured text and return content from each main section.

    Extracts non-header text from sections and subsections,
    grouping everything under the parent main section (1., 2., 3., etc.).
    """
    # Pattern matches lines like "1. Title", "1.1 Title", "2.3.1 Title"
    header_pattern = re.compile(r'^(\d+(?:\.\d+)*)\.?\s+')

    result = []
    current_content_lines = []
    in_section = False

    for line in text.split('\n'):
        stripped = line.strip()
        match = header_pattern.match(stripped)

        if match:
            section_num = match.group(1)
            # Check if this is a main section (single number, e.g. "1", "2")
            is_main = '.' not in section_num

            if is_main:
                # Save previous main section's content
                if in_section:
                    result.append(' '.join(current_content_lines))
                current_content_lines = []
                in_section = True
        elif stripped and in_section:
            # Non-empty, non-header line — collect as content
            current_content_lines.append(stripped)

    # Don't forget the last section
    if in_section:
        result.append(' '.join(current_content_lines))

    # Exclude sections with no content
    return [s for s in result if s]


if __name__ == '__main__':
    text1 = """
1. Вступление
Описание курса, начальные требования, для кого этот курс.
1.1 Как правильно входить в курс
Цели курса, канал и комьюнити, глоссарий, PET-проект, хард режим vs лайт режим?
1.2 Общий подход и точки улучшения приложений с LLM
Рассказываем почему важно разбираться в LLM. Нужен ли ИИ обычному человеку?
1.3 API ключ курса или от OpenAI?
1.3.1 Ключ от команды курса
Получаем ключ в боте.
1.3.2 Ключ от OpenAI
Получаем официальный ключ от OpenAI.
1.3.3 Ключ от HuggingFace
Ныряем в Open Source и получаем ключ от HuggingFace.
2. Промптинг - объясни LLM, что тебе от неё надо!
2.1 Введение в Prompt Engineering
Поясняем за промпты. Техники и лайфхаки для промптинга. Из чего состоит промпт?
2.2 Дизайн промптов в LangChain
Few-shot learning. Output Parser.
2.2.1 Введение в LangChain
Рассказываем про преимущества LangChain.
3. LangChain или причем тут попугаи?
3.1 Память в LangChain
Переводим LLM в чат-режим. Типы памяти.
3.2 Chains - собери свою цепь
Chains & LCEL.
3.3 Агенты intro
Агенты и цепи. Инструменты (tools).
"""

    text2 = """


"""

    text3 = """
1. Главный заголовок
Тут содержимое, которое нужно достать из главного заголовка.
1.1 Подзаголовок
Тут содержимое, которое нужно достать из подзаголовка.
"""

    text4 = """
1. Первый заголовок

2. Второй заголовок
"""

    text5 = """
1. Введение
Краткое описание целей и задач документа.

2. Основные разделы
Краткий обзор основных разделов.

2.1 Первый подраздел
Какие-то мелкие детали.

3. Заключение
Выводы и заключительные комментарии.
4. Приложения
Дополнительная информация и материалы.
"""

    for i, text in enumerate([text1, text2, text3, text4, text5], 1):
        sections = get_content_from_sections(text)
        print(f"text{i}: {sections}")
        print(f"  length: {len(sections)}")
        print()
