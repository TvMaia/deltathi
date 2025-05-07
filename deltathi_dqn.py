import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
from deltathi_api import DeltathiAPI  # Importando os deltas da API de trade

# Configurações
MAX_DELTAS = 20
STATE_SIZE = MAX_DELTAS + 1  # Deltas + pontuação
ACTION_SIZE = 3  # Subir (buy), descer(sell), finalizar (take profit)
EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.95  # Fator de desconto
EPSILON = 1.0  # Exploração inicial
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001

class DQNAgent:
    def __init__(self):
        self.memory = deque(maxlen=2000)
        self.model = self._build_model()
        self.epsilon = EPSILON

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=(STATE_SIZE,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(ACTION_SIZE, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(ACTION_SIZE)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + GAMMA * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

def train_dqn():
    agent = DQNAgent()
    for e in range(EPISODES):
        api = DeltathiAPI()  # Instância do simulador
        deltas = []
        points = 0
        state = np.zeros(STATE_SIZE)
        total_reward = 0
        steps = 0

        print(f"\n=== Episódio {e+1}/{EPISODES} ===")

        while not api.is_over():
            steps += 1
            # Obter próximo delta
            delta = api.get_next_delta()
            deltas = api.get_deltas()

            # Estado: deltas + pontuação atual
            state = np.array(deltas + [0] * (MAX_DELTAS - len(deltas)) + [points]).reshape(1, -1)

            # Escolher ação
            action_idx = agent.act(state)
            action = ['subir', 'descer', 'finalizar'][action_idx]

            # Executar ação e obter recompensa
            reward = api.perform_action(action)
            next_points = api.get_points()

            # Recompensa baseada na variação dos pontos
            reward = next_points - points if action != 'finalizar' else reward
            points = next_points

            # Próximo estado
            next_deltas = api.get_deltas()
            next_state = np.array(next_deltas + [0] * (MAX_DELTAS - len(next_deltas)) + [points]).reshape(1, -1)

            # Salvar experiência
            agent.remember(state, action_idx, reward, next_state, api.is_over())

            total_reward += reward

            # Mostrar progresso no console
            if steps % 10 == 0:
                print(f"Passo {steps}: Ação = {action}, Pontos = {points}, Recompensa = {reward}")

        # Treinar com replay
        agent.replay()

        print(f"Episódio {e+1}/{EPISODES}: Recompensa total = {total_reward}, Pontos finais = {points}")

    # Salvar o modelo treinado
    agent.model.save('deltathi_dqn_model.h5')
    print("Modelo salvo em 'deltathi_dqn_model.h5'")

if __name__ == "__main__":
    train_dqn()