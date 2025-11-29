LEGENDA-ME: Software de Tradução e Legendagem de Áudio com Inteligência Artificial

Visão Geral do Projeto:

O LEGENDA-ME é um Software de Tradução e Legendagem de Áudio com Inteligência Artificial desenvolvido como Trabalho de Conclusão de Curso (TCC) em Engenharia de Software. O projeto visa superar as barreiras linguísticas, oferecendo uma solução web robusta e acessível para transcrição e tradução de áudio capturado pelo microfone.

A principal inovação reside na integração de modelos de Inteligência Artificial de código aberto que operam localmente no servidor (backend), eliminando a dependência de APIs comerciais e garantindo a privacidade dos dados do usuário e um baixo custo operacional.

Status do Projeto

Este projeto é um Produto Mínimo Viável (MVP) validado, conforme detalhado no TCC.

Funcionalidades Principais:

Captura de Áudio: Utiliza a Web Audio API para capturar o áudio do microfone diretamente no navegador.

Transcrição de Baixa Latência: Emprega o modelo Faster-Whisper (versão otimizada do OpenAI Whisper) para conversão de fala em texto com alta performance.

Tradução Neural Local: Utiliza o modelo MarianMT (Hugging Face) para tradução automática neural, garantindo que todo o processamento de IA ocorra no servidor.

Acessibilidade Web: Construído sobre o framework Django, o software é acessível.

Tecnologias Utilizadas:

Python

Django - Framework web robusto para orquestração do sistema

ASR (Transcrição)

Faster-Whisper

Reconhecimento Automático de Fala

NMT (Tradução)

MarianMT

Modelo de Tradução Automática Neural

IA/Deep Learning

PyTorch 

Hugging Face Transformers

Frontend:
HTML5, CSS3, JavaScript (Web Audio API)
Interface de usuário e captura de áudio no navegador.


Configuração e Instalação:

Este guia pressupõe que você tenha o Python e o pip instalados em seu sistema.

1. Clonar o Repositório

```bash
git clone [https://github.com/DaniloFloridi/Legenda-me]
cd transcricao_web-main
```

2. Configurar o Ambiente Virtual

É altamente recomendável usar um ambiente virtual para isolar as dependências:

```bash
python3 -m venv venv
source venv/bin/activate  # No Linux/macOS
# venv\Scripts\activate   # No Windows
```

3. Instalar Dependências

As principais bibliotecas necessárias para o projeto são:


pip install django faster-whisper transformers torch


4. Configuração dos Modelos de IA:

O projeto utiliza modelos de IA que precisam ser carregados localmente.

#### 4.1. Modelo Faster-Whisper

O modelo medium será baixado automaticamente na primeira execução.

#### 4.2. Modelo MarianMT para Tradução

O código-fonte (audio_app/views.py) espera o modelo de tradução Inglês para Português (opus-mt-en-pt) em um caminho específico.

**Atenção:** O caminho no código (r"E:\Projeto Atualizado - Django Trial\models\opus-mt-en-pt") é um caminho local do desenvolvedor e **precisa ser ajustado**.

a. Baixe o modelo: O modelo pode ser baixado e carregado automaticamente pelo Hugging Face, mas para garantir o funcionamento local, você pode usar o código abaixo para baixá-lo e salvar em um diretório:

```python
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-pt"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Crie um diretório para salvar o modelo, por exemplo, 'models/opus-mt-en-pt'
save_directory = "./models/opus-mt-en-pt" 
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
```

b. Atualize o caminho no views.py: Edite o arquivo audio_app/views.py (linha 15) para apontar para o diretório onde você salvou o modelo:

```python
marian_model_name = "./models/opus-mt-en-pt" # Ou o caminho que você escolheu
```

5. Executar o Servidor Django

Com as dependências instaladas e o caminho do modelo ajustado, você pode iniciar o servidor de desenvolvimento:


python manage.py runserver


O aplicativo estará acessível em http://127.0.0.1:8000/.

Arquitetura do Sistema:

O sistema segue o padrão Model-View-Template (MVT) do Django e é dividido em três módulos principais:

1. Frontend (Navegador): Responsável pela interface e pela captura de áudio. O JavaScript utiliza a Web Audio API para gravar o áudio do microfone e enviá-lo em *chunks* via requisição `POST` para o backend.

2. Backend (Django): Recebe o áudio, gerencia o fluxo de processamento e coordena os módulos de IA.

3. Módulos de IA (Local):

Transcrição: O áudio é processado pelo Faster-Whisper (modelo definido pelo usuário), que o converte em texto (transcrição).

Tradução: O texto transcrito é imediatamente traduzido pelo MarianMT (opus-mt-en-pt) para o idioma alvo.


O backend retorna a transcrição e a tradução em uma resposta JSON, que é então exibida na interface do usuário.


Referência:

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) para o curso de Engenharia de Software do Centro Universitário de Ourinhos.

Título: LEGENDA-ME - SOFTWARE DE TRADUÇÃO E LEGENDAGEM DE ÁUDIO COM INTELIGÊNCIA ARTIFICIAL Autores: ANNA LAURA DA COSTA POLO, DANIEL GUIMARÃES DE FARIA, DANILO FRANCISCO MARTINS FLORIDI, SILVIO JALES ROSA JUNIOR Ano: 2025
