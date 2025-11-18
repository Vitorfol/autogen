# -*- coding: utf-8 -*-
"""
AutoGen CLI - Execute tarefas com IA via linha de comando

Exemplos de uso:
  # Usando argumentos diretamente:
  python run_chat.py --task "Write a hello world in Python" --api-key sk-xxx --model gpt-4o-mini
  
  # Usando variáveis de ambiente:
  export OPENAI_API_KEY=sk-xxx
  python run_chat.py --task "Explain quantum computing" --model gpt-4o-mini
  
  # Modo interativo (pergunta as informações):
  python run_chat.py
"""
import asyncio
import argparse
import os
import sys
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


def get_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Execute tarefas com AutoGen via linha de comando",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s --task "Explain AI" --model gpt-4o-mini
  %(prog)s --task "Write Python code" --api-key sk-xxx --model gpt-3.5-turbo
  export OPENAI_API_KEY=sk-xxx && %(prog)s --task "Help me"
        """
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        help="Tarefa que você deseja que o assistente execute"
    )
    
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=os.environ.get("OPENAI_API_KEY"),
        help="Chave de API da OpenAI (ou use a variável de ambiente OPENAI_API_KEY)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-5-mini",
        help="Modelo a ser usado (padrão: gpt-5-mini)"
    )
    
    parser.add_argument(
        "--system-message", "-s",
        type=str,
        default="You are a helpful AI assistant. Solve tasks carefully.",
        help="Mensagem de sistema para o assistente"
    )
    
    return parser.parse_args()


async def main():
    """Executa a tarefa usando o assistente."""
    args = get_args()
    
    # Modo interativo se os argumentos não foram fornecidos
    if not args.task:
        print("🤖 AutoGen CLI - Modo Interativo\n")
        args.task = input("📝 Qual tarefa você deseja executar? ")
        if not args.task.strip():
            print("❌ Erro: Tarefa não pode estar vazia!")
            sys.exit(1)
    
    if not args.api_key:
        args.api_key = input("🔑 Digite sua chave de API da OpenAI: ")
        if not args.api_key.strip():
            print("❌ Erro: API key é obrigatória!")
            print("💡 Dica: Defina a variável de ambiente OPENAI_API_KEY ou use --api-key")
            sys.exit(1)
    
    print(f"\n🚀 Iniciando tarefa: {args.task}")
    print(f"🤖 Modelo: {args.model}\n")
    
    # Crie o cliente do modelo
    model_client = OpenAIChatCompletionClient(
        model=args.model,
        api_key=args.api_key,
    )
    
    # Cria o Agente Assistente
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message=args.system_message,
    )
    
    try:
        # Executa a tarefa e obtém o resultado
        result = await assistant.run(task=args.task)
        
        # Imprime as mensagens da conversa
        print("\n" + "="*60)
        print("📋 RESULTADO")
        print("="*60)
        for message in result.messages:
            print(f"\n[{message.source.upper()}]:")
            print(message.content)
            print("-" * 60)
        
    except Exception as e:
        print(f"\n❌ Erro ao executar tarefa: {e}")
        sys.exit(1)
    finally:
        # Fecha o cliente quando terminar
        await model_client.close()
    
    print("\n✅ Tarefa concluída!")


# Executa o programa
if __name__ == "__main__":
    asyncio.run(main())