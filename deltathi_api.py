import random

class DeltathiAPI:
    def __init__(self):
        self.deltas = []  # Lista de deltas passados
        self.points = 0   # Pontuação atual
        self.game_over = False
        self.obstacle_proximity = random.randint(20, 80)  # Simula proximidade do obstáculo

    def get_next_delta(self):
        # Simula um novo delta baseado em padrões de obstáculos
        delta = random.randint(1, 10)
        self.deltas.append(delta)
        if len(self.deltas) > 20:  # Limita a 20 deltas
            self.deltas.pop(0)
        return delta

    def perform_action(self, action):
        if action == 'finalizar':
            self.game_over = True
            return self.points if self.points > 0 else -1000  # Ganha pontos positivos ou perde com negativos
        else:
            # Ajusta proximidade com base na ação
            if action == 'subir':
                self.obstacle_proximity -= random.randint(3, 7)
            elif action == 'descer':
                self.obstacle_proximity += random.randint(3, 7)

            # Verifica colisão
            if self.obstacle_proximity <= 0:
                self.game_over = True
                return -1000  # Perde ao colidir

            # Atualiza pontuação com base na proximidade
            if self.obstacle_proximity > 50:
                self.points += 10  # Longe: ganha mais pontos
            elif self.obstacle_proximity > 20:
                self.points += 5   # Médio: ganha menos
            else:
                self.points -= 10  # Perto: perde pontos

            return self.points

    def is_over(self):
        return self.game_over

    def get_points(self):
        return self.points

    def get_deltas(self):
        return self.deltas.copy()