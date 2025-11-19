# -*- coding: utf-8 -*-
"""
AutoGen CLI - Execute tarefas com IA via linha de comando

Exemplos de uso:
  # Usando argumentos diretamente:
  python run_chat.py --task "Write a hello world in Python" --api-key sk-xxx --model gpt-4o-mini
  
  # Usando variáveis de ambiente:
  export OPENAI_API_KEY=sk-xxx
  python run_chat.py --task "Explain quantum computing" --model gpt-4o-mini
  
  # Com execução de código:
  python run_chat.py --task "Create a Python script that calculates fibonacci" --execute-code
  
  # Modo interativo (pergunta as informações):
  python run_chat.py
"""
import asyncio
import argparse
import os
import sys
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor


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
    
    parser.add_argument(
        "--execute-code", "-e",
        action="store_true",
        help="Habilitar execução automática de código gerado"
    )
    
    parser.add_argument(
        "--work-dir", "-w",
        type=str,
        default="coding",
        help="Diretório onde o código será salvo e executado (padrão: coding)"
    )
    
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Número máximo de turnos de conversação entre agentes (padrão: 20)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout de execução de código em segundos (padrão: 120)"
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
    print(f"🤖 Modelo: {args.model}")
    if args.execute_code:
        print(f"⚙️  Execução de código: Habilitada")
        print(f"📁 Diretório de trabalho: {args.work_dir}")
        print(f"🔄 Max turnos: {args.max_turns}")
        print(f"⏱️  Timeout: {args.timeout}s")
    print()
    
    # Crie o cliente do modelo
    model_client = OpenAIChatCompletionClient(
        model=args.model,
        api_key=args.api_key,
    )
    
    try:
        if args.execute_code:
            # Modo com execução de código
            print("🔧 Configurando agentes para execução de código...\n")
            
            # Cria o assistente que gera código
            assistant = AssistantAgent(
                name="assistant",
                model_client=model_client,
                system_message=f"{args.system_message} You MUST write executable code using markdown code blocks (```python). The code will be automatically executed. Wait to see the execution result before saying TERMINATE. Only reply with TERMINATE after you have seen the code execution output.",
            )
            
            # Cria o executor de código (configurado para manter os arquivos)
            code_executor = LocalCommandLineCodeExecutor(
                work_dir=args.work_dir,
                cleanup_temp_files=False,  # NÃO deletar arquivos após execução
                timeout=args.timeout,  # Timeout configurável via CLI
            )
            executor_agent = CodeExecutorAgent(
                name="code_executor",
                code_executor=code_executor,
            )
            
            # Define condições de terminação
            # 1. Para quando o assistente diz "TERMINATE" (tarefa completa)
            # 2. Para quando atingir max_turns (safety net)
            text_termination = TextMentionTermination("TERMINATE")
            max_turns_termination = MaxMessageTermination(max_messages=args.max_turns)
            
            # Usa ambas as condições (OR logic - para na primeira que acontecer)
            termination = text_termination | max_turns_termination
            
            # Cria um grupo com os dois agentes
            team = RoundRobinGroupChat(
                participants=[assistant, executor_agent],
                termination_condition=termination,
                max_turns=args.max_turns,  # Mantém como fallback extra
            )
            
            # Executa a tarefa com o time
            print("="*60)
            print("💬 CONVERSAÇÃO")
            print("="*60 + "\n")
            
            stream = team.run_stream(task=args.task)
            await Console(stream)
            
            print("\n" + "="*60)
            print(f"✅ Código salvo e executado em: {args.work_dir}/")
            print("="*60)
            
        else:
            # Modo simples sem execução de código
            assistant = AssistantAgent(
                name="assistant",
                model_client=model_client,
                system_message=args.system_message,
            )
            
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