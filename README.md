Filipe... fiz correndo só pra ver aqui a brisa.

Fiz esse código para te mostrar minha ideia de usar Reinforcement Learning pra operar trades de forma automática. Olha aqui, mano:

🚀 Visão Geral

Objetivo: Treinar um agente de Deep Q-Network (DQN) pra decidir quando comprar (subir), vender (descer) ou fechar posição (finalizar) com base em variações de preço (deltas).

Por quê?: Em vez de regras fixas, o agente aprende pela experiência, acumulando ganhos e minimizando perdas ao longo de vários ciclos de simulação.

💡 Como Funciona o Código

Ambiente Simulado (DeltathiAPI): fornece deltas de preço sequenciais e recebe ações do agente.

Estado: último conjunto de MAX_DELTAS (20) + pontuação atual. É um vetor de tamanho STATE_SIZE = 21.

Ações (ACTION_SIZE = 3): subir, descer, finalizar.

Recompensa: varia conforme a diferença de pontos antes e depois da ação; garante que o agente foque em maximizar retornos.

Replay Buffer: guarda até 2.000 experiências (state, action, reward, next_state, done) para treinar em minibatches de 32 experiências.

Treino: usa ε-greedy para equilibrar exploração/exploração e atualiza a rede neural via otimização MSE.

🔍 Principais Diferenciais e Eficiência

Aprendizado Contínuo: cada episódio ajusta o modelo, acumulando conhecimento de cenários diversos.

Replay Buffer: evita correlações fortes entre experiências consecutivas, aumentando estabilidade do aprendizado.

Arquitetura Simplicidade-Versatilidade: duas camadas densas com 64 neurônios cada garantem bom trade-off entre complexidade e performance.

ε-decay automático: reduz gradualmente a aleatoriedade, focando em segurança e consistência conforme o modelo amadurece.

Visão de futuro: dá pra turbinar com target network, batch training vetorizado e métricas de risco extras (drawdown, sharpe, etc.).

⚙️ Estrutura do Repositório
├── deltathi_dqn_model.h5    # Modelo final salvo
├── train_dqn.py             # Script principal de treino e simulação
└── requirements.txt         # NumPy, TensorFlow, DeltathiAPI