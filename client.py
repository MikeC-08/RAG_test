import scripts.llm as llm
import scripts.rag as rag

def main():
    model = rag.load_model()
    rag.embed_content(model)
    while True:
        client_message = input("> ")
        rag_txts = rag.get_rag_embedding(client_message, model)
        result = llm.request(client_message, rag_txts)
        print(result)

if __name__ == '__main__':
    main()