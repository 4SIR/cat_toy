
<div align="center">
  <h1>Cat Toy</h1>
  <p>Um brinquedo para gatos para pessoas preguiçosas.</p>
  <img src="https://img.shields.io/github/license/ThomasBouasli/cat_toy" alt="License">
  <img src="https://github.com/ThomasBouasli/cat_toy/actions/workflows/lint.yaml/badge.svg" alt="Lint Badge">
</div>

## Descrição

O Cat Toy é um brinquedo para gatos que utiliza um laser para entreter o gato. O brinquedo é controlado por um raspberry pi que identifica o gato na camera e move o laser para que o gato se mova.

## Instalação

Para instalar o projeto, você precisa clonar o repositório e instalar as dependências.

```bash
git clone https://github.com/4SIR/cat_toy.git
cd cat_toy
pip install -r requirements.txt
```

## Uso

Para rodar o projeto, você precisa rodar o arquivo `main.py`.

```bash
python main.py
```

## Definição de Pronto

- [ ] Identificar um gato na camera (1,0)
- [ ] Identificar a posição do gato na camera (2,5)
- [ ] Determinar o vetor de movimento do gato (2,5)
- [ ] Determinar para onde mover o laser para que o gato se mova (2,0)
- [ ] Implementar o código num raspberry pi e fazer o hardware (2,0)

