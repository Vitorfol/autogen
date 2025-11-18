# -*- coding: utf-8 -*-
"""
Este é um script editável para executar conversas com o AutoGen via CLI.

Para usar:
1. Altere o valor da variável 'minha_tarefa' abaixo para o que você deseja que os agentes façam.
2. Altere a chave de API e o modelo se necessário.
3. Execute o script no terminal: python run_chat.py
"""
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

# =================================================================================
# PARTE EDITÁVEL: Defina sua tarefa aqui
# =================================================================================
minha_tarefa = "Prompt"
# =================================================================================
# FIM DA PARTE EDITÁVEL
# =================================================================================

print(f"Iniciando a tarefa: {minha_tarefa}\n")

# Configure sua chave de API e modelo aqui
api_key = "your-key-here"
model = "gpt-5-mini"  # Modelo mais acessível. Outras opções: "gpt-3.5-turbo", "gpt-4o"

# Crie o cliente do modelo
model_client = OpenAIChatCompletionClient(
    model=model,
    api_key=api_key,
)

# Cria o Agente Assistente
assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a helpful AI assistant. Solve tasks carefully.",
)


async def main():
    """Executa a tarefa usando o assistente."""
    # Executa a tarefa e obtém o resultado
    result = await assistant.run(task=minha_tarefa)
    
    # Imprime as mensagens da conversa
    print("\n=== RESULTADO ===")
    for message in result.messages:
        print(f"\n[{message.source}]: {message.content}")
    
    # Fecha o cliente quando terminar
    await model_client.close()
    print("\n\nTarefa concluída!")


# Executa o programa
if __name__ == "__main__":
    asyncio.run(main())