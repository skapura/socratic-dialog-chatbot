import ragchatbot as rag

#rag.reindex_database()

prompt, memory, llm, retriever = rag.make_pipeline()

bot = rag.RagChatbot(prompt, memory, llm, retriever)

while True:

    print('-----------------------')
    user_input = input("Enter your question (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    response = bot.converse(user_input)

    if response['docs']:
        print('Documents:')
        for d in response['docs']:
            print(f"{d['doc']} - score: {d['score']}")

    print('Response:')
    print(response['response'])