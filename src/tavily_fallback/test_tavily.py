from tavily_client import TavilySearchClient
from tavily_service import TavilyService

def main():
    service = TavilyService(TavilySearchClient())
    result = service.get_answer("My father fell from stairs and broke his hip. He's 68 years old. The policy was taken last week only. Emergency case. Will insurance pay?")

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nSOURCES:")
    for src in result["sources"]:
        print("-", src)

if __name__ == "__main__":
    main()
