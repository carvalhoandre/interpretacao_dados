import turtle

'''

Parâmetros que delimitam o labirinto, indicam os obstaculos, caminhos livres para seguir, saida do labirinto e caminho correto identificado.

PART_OF_PART - O caminho correto é sinalizado retornando ao ponto de partida.
TRIED -  Caminho percorrido pelo agente. Sinaliza o caminho que ele esta buscando pela saida. 
OBSTACLE - O caminho contém obstaculos que delimitam o labirinto e são representados pelo simbolo +.
DEAD_END - Sinaliza caminhos que o agente já percorreu e estão errados. 

'''
PART_OF_PATH = 'O'
TRIED = '.'
OBSTACLE = '+'
DEAD_END = '-'


class Maze:
    '''
    A função __init__ lê o arquivo com a matriz que representa o labirinto, lê a quantidade de linhas e colunas, bem como linha a coluna de inicio
    Instância o Turtle para gerar a interface gráfica e utiliza como coordenadas as linhas e colunas da nossa matriz
    A posição inicial do agente é lida através do loop na função.
    Instanciamos o turtle, definimos um formato do agente que pode ser turtle, arrow, circle, square, triangle, classic.

    '''

    def __init__(self, mazeFileName):
        rowsInMaze = 0
        columnsInMaze = 0
        self.mazelist = []
        mazeFile = open(mazeFileName, 'r')
        rowsInMaze = 0
        for line in mazeFile:
            rowList = []
            col = 0
            for ch in line[:-1]:
                rowList.append(ch)
                if ch == 'S':
                    self.startRow = rowsInMaze
                    self.startCol = col
                col = col + 1
            rowsInMaze = rowsInMaze + 1
            self.mazelist.append(rowList)
            columnsInMaze = len(rowList)

        self.rowsInMaze = rowsInMaze
        self.columnsInMaze = columnsInMaze
        self.xTranslate = -columnsInMaze / 2
        self.yTranslate = rowsInMaze / 2
        self.t = turtle.Turtle()
        self.t.shape('turtle')
        turtle.title('Desafio saida de labirinto')
        self.wn = turtle.Screen()
        self.wn.setworldcoordinates(-(columnsInMaze - 1) / 2 - .5, -(rowsInMaze - 1) / 2 - .5,
                                    (columnsInMaze - 1) / 2 + .5, (rowsInMaze - 1) / 2 + .5)

    def drawMaze(self):
        '''
        Função que cria a interação do gráfico do labirinto, temos a velocidade, o tracer, criamos uma lista com a linha e coluna
        checamos se é um obstáculo e pintamos de laranja para gerar o mapa do labirinto.
        O rastro do agente é da cor cinza e o agente da cor vermelho e pode ser alterado nas configurações abaixo.
        '''
        self.t.speed(10)
        self.wn.tracer(0)
        for y in range(self.rowsInMaze):
            for x in range(self.columnsInMaze):
                if self.mazelist[y][x] == OBSTACLE:
                    self.drawCenteredBox(x + self.xTranslate, -y + self.yTranslate, 'orange')
        self.t.color('gray')
        self.t.fillcolor('red')
        self.wn.update()
        self.wn.tracer(1)

    def drawCenteredBox(self, x, y, color):

        '''
        Esta função recebe coluna, linha e cor que será aplicada para o centro do box.
        '''
        self.t.up()
        self.t.goto(x - .5, y - .5)
        self.t.color(color)
        self.t.fillcolor(color)
        self.t.setheading(90)
        self.t.down()
        self.t.begin_fill()
        for i in range(4):
            self.t.forward(1)
            self.t.right(90)
        self.t.end_fill()

    def moveAgent(self, x, y):

        '''
        Função que move o agente, a chamada "goto" faz o movimento do agente.
        '''
        self.t.up()
        self.t.setheading(self.t.towards(x + self.xTranslate, -y + self.yTranslate))
        self.t.goto(x + self.xTranslate, -y + self.yTranslate)

    def dropBreadcrumb(self, color):
        self.t.dot(10, color)

    def updatePosition(self, row, col, val=None):

        '''
        Checa se a posição indicada é valida e movimenta o agente para nova posição, Se a posição é valida a cor azul é aplicada ao rastro,
        Se for um caminho já explorado a cor vermelha é aplicada, caso tenha finalizado a saida o percurso de volta é salvo em verde.
        '''

        if val:
            self.mazelist[row][col] = val
        self.moveAgent(col, row)

        if val == PART_OF_PATH:
            color = 'green'
        elif val == OBSTACLE:
            color = 'red'
        elif val == TRIED:
            color = 'blue'
        elif val == DEAD_END:
            color = 'red'
        else:
            color = None

        if color:
            self.dropBreadcrumb(color)

    def isExit(self, row, col):
        '''
        Função de saida de acordo com as regras da matriz 0 ou rowsInMaze-1 determinam a saida.
        '''

        return (row == 0 or
                row == self.rowsInMaze - 1 or
                col == 0 or
                col == self.columnsInMaze - 1)

    def __getitem__(self, idx):
        return self.mazelist[idx]


def searchFrom(maze, startRow, startColumn):
    '''
    Função de busca em si, recebe a matriz (maze) linha e coluna de inicio. Aqui aplicamos os testes de direção e vamos explorando o caminho
    usando as demais funções.
    '''
    # Tente cada uma das posições até encontrar a saida
    # Valores de retorno na saida da base
    #  1. Se encontrar um obstaculo retornar false
    maze.updatePosition(startRow, startColumn)
    if maze[startRow][startColumn] == OBSTACLE:
        return False
    #  2. Encontrou uma área que já foi explorada
    if maze[startRow][startColumn] == TRIED or maze[startRow][startColumn] == DEAD_END:
        return False
    # 3. Encontrou uma borda não ocupada por um obstáculo
    if maze.isExit(startRow, startColumn):
        maze.updatePosition(startRow, startColumn, PART_OF_PATH)
        return True
    maze.updatePosition(startRow, startColumn, TRIED)
    print(startColumn, startRow)
    # Caso contrário teste cada direção novamente
    found = searchFrom(maze, startRow - 1, startColumn) or \
            searchFrom(maze, startRow + 1, startColumn) or \
            searchFrom(maze, startRow, startColumn - 1) or \
            searchFrom(maze, startRow, startColumn + 1)
    if found:
        maze.updatePosition(startRow, startColumn, PART_OF_PATH)
    else:
        maze.updatePosition(startRow, startColumn, DEAD_END)
    return found

myMaze = Maze('D:\Users\andre\Documents\Faculdade\inteligencia artificial\maze2.txt')
myMaze.drawMaze()
myMaze.updatePosition(myMaze.startRow,myMaze.startCol)

searchFrom(myMaze, myMaze.startRow, myMaze.startCol)