# VoltLens — Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet

**Disciplina:** Visão Computacional — CESAR School, 2026.2  
**Integrantes:**   
· Helder Barros (hcbf@cesar.school)  
· Felipe Salustiano 2 (fsm2@cesar.school)  
· Lucas Mendes (ljm@cesar.school)  
· Lucas Rafael (lrgms@cesar.school)  
· José Kevin (jkss@cesar.school)  

## Fonte dos dados e licença

Dataset: [Yandex.Toloka Water Meters](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset),
licença **CC BY-NC-ND 4.0**. Este repositório **não** contém imagens do dataset, recortes gerados nem checkpoints.
O uso é restrito a fins acadêmicos e não comerciais.

## Como reproduzir

```bash
python -m venv .venv
.venv\Scripts\activate                       # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU NVIDIA (RTX 50xx exige CUDA 12.8)
pip install -r requirements.txt
python -m ipykernel install --user --name vc-p2-hidrometros
# Dados: o notebook procura data.csv em ../vc-20262-p2/data (ou na variável WATERMETERS_DIR).
# Se não encontrar, baixa pela API do Kaggle (kagglehub) -> precisa de ~/.kaggle/kaggle.json.
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 pratica_watermeters.ipynb
python tools/clear_image_outputs.py pratica_watermeters.ipynb   # limpa as saídas que exibem imagens do dataset
```

- Ambiente usado: local, Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, torchvision 0.26.0, GPU **NVIDIA GeForce RTX 5060 Ti (8 GB)**
- Tempo total de execução: ~15 min (10,5 min de treino dos quatro modelos)
- Split utilizado: `splits.csv` **oficial** fornecido pelo professor (não alterado)
- As células que exibem imagens do dataset têm a tag `dataset-images` e a saída limpa antes do commit.

| Arquivo | Conteúdo |
|---|---|
| `pratica_watermeters.ipynb` | notebook executado (Partes 1–5) |
| `splits.csv` | divisão oficial (cópia sem alteração) |
| `results.csv`, `history_*.csv` | métricas no teste e histórico de treino |
| `figures/` | 4 figuras obrigatórias + 2 da análise exploratória |
| `reports/` | `classification_report` de cada modelo, latência em CPU, orientação de cada faixa, tipos de erro |

## Protocolo de treino

| Modelo | Pesos | Entrada | Batch | LR cabeça | LR backbone | Épocas máx. | Melhor época |
|---|---|---|---|---|---|---|---|
| smallcnn | do zero | 64×48 | 256 | 1e-3 | — | 10 | 10 |
| alexnet | ImageNet (IMAGENET1K_V1) | 128×128 | 128 | 1e-3 | 1e-4 | 8 | 6 |
| vgg16_bn | ImageNet (IMAGENET1K_V1) | 128×128 | 32 | 1e-3 | 1e-5 | 8 | 7 |
| resnet18 | ImageNet (IMAGENET1K_V1) | 128×128 | 128 | 1e-3 | 1e-4 | 8 | 7 |

Comum a todos: `AdamW` (weight decay 0,01) + `OneCycleLR` (passo por *batch*, `max_lr` = LR de cada grupo);
`CrossEntropyLoss(label_smoothing=0.1)`; 1ª época com o backbone congelado (pré-treinados); *early stopping* pelo
F1-macro da validação, paciência 3 (nenhum modelo chegou a parar antes); `seed=42` com cuDNN determinístico;
*augmentation* `RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))` +
`ColorJitter(brightness=0.3, contrast=0.3)`, sem flip; normalização com média/desvio do ImageNet.

Alterações em relação ao protocolo oficial (e justificativa): **nenhuma alteração de hiperparâmetro.** Decisões de
implementação, iguais para os quatro modelos:

- **Cabeça** = a última camada linear (substituída por `Linear(·, 10)`); **backbone** = todo o resto. Na `smallcnn`
  todos os parâmetros usam LR 1e-3.
- A *augmentation* é aplicada no recorte original 64×48; o redimensionamento bilinear para 128×128 e a normalização
  são feitos na GPU (mais rápido que na CPU no Windows, onde usamos `num_workers=0`).
- `seconds` no histórico = tempo da época de treino, **sem** a validação; `s_per_epoch` = mediana dessa coluna.
- `ms_per_image` = latência na GPU, batch 1, entrada já pré-processada, mediana de 200 execuções após 20 de *warm-up*.

## Resultados (conjunto de teste)

| model | acc_digit | f1_macro | acc_reading | params_M | s_per_epoch | ms_per_image | ckpt_MB | best_epoch |
|---|---|---|---|---|---|---|---|---|
| smallcnn | 0.8008 | 0.7522 | 0.2941 | 0.928 | 5.03 | 0.982 | 3.55 | 10 |
| alexnet | 0.9171 | 0.8968 | 0.6257 | 57.045 | 5.48 | 1.347 | 217.61 | 6 |
| vgg16_bn | 0.9372 | 0.9283 | 0.7112 | 134.31 | 29.19 | 3.325 | 512.41 | 7 |
| resnet18 | 0.9599 | 0.9536 | 0.8449 | 11.182 | 6.94 | 3.02 | 42.73 | 7 |

Latência adicional em **CPU** (batch 1, mesma metodologia; `reports/cpu_latency.csv`): smallcnn 2,42 ms ·
alexnet 9,94 ms · vgg16_bn 40,51 ms · resnet18 9,36 ms.

Figuras (pasta `figures/`):

- Matrizes de confusão normalizadas: `figures/confusion_matrices.png`
- Acurácia por posição: `figures/accuracy_by_position.png`
- Curvas de treino: `figures/training_curves.png`
- Desempenho × custo: `figures/performance_vs_cost.png`
- Extras da análise exploratória: `figures/eda_readings.png`, `figures/eda_digit_distribution.png`

## Respostas às questões

**1.1** O dataset tem três pastas com os mesmos 1244 nomes de arquivo e um `data.csv`:
`images/` traz as fotos originais dos hidrômetros (JPEG, 1000–2057 px; 288 têm tag EXIF de rotação), `masks/` traz a
máscara binária da região do visor e `collage/` traz a foto com a máscara sobreposta (só para visualização).
No `data.csv`, `photo_name` é o nome do arquivo, `value` é a leitura em m³ (float) e `location` é o polígono do visor
em coordenadas normalizadas (0–1), com 4, 6 ou 8 vértices. Para treinar servem `images/` (entrada) e `data.csv`
(`value` → rótulo, `location` → onde recortar). `masks/` é redundante com o polígono (IoU mediana 0,983, mínima
0,936, diferença de rasterização) e `collage/` não acrescenta informação.

**1.2** O rótulo é **por imagem**: há apenas a leitura completa, sem a posição de cada dígito. Para classificar
dígitos precisamos inferir onde cada um está: o pipeline divide a faixa retificada em 8 partes iguais e atribui o
*i*-ésimo caractere do rótulo ao *i*-ésimo recorte. Isso é uma supervisão fraca: qualquer erro de recorte
(orientação, desalinhamento, polígono frouxo) vira **ruído de rótulo** no treino e erro "falso" no teste, e a leitura
só fica correta se os 8 recortes estiverem corretos.

**2.1** O **nome do arquivo**, interpretado como texto. Ele coincide com `value` em 1244/1244 imagens, mas `value` é
um float: o CSV grava `80.19` para um visor que mostra `00080,190`. Há 41 leituras com 1 casa decimal e 87 com 2.
Conferimos visualmente amostras desses casos e o visor sempre tem 3 decimais, com o zero final. Portanto o
rótulo de 8 dígitos é `inteiro.zfill(5) + decimal.ljust(3, '0')`. A string do nome também deixa explícita a
única leitura fora do padrão (`id_664_value_45_2676.jpg`, 4 decimais), que foi descartada.

**2.2** O **0**, com 35,7 % de todos os dígitos (o segundo, o 1, tem 10,1 %). 95,9 % das leituras têm parte inteira
com até 3 dígitos (mediana 188,8 m³), então as posições 0 e 1 são quase sempre zeros à esquerda. Além disso, as 128
leituras com menos de 3 decimais ganham zeros à direita. No teste, 34,4 % dos recortes são 0.

**3.1** Os 8 recortes de uma imagem compartilham hidrômetro, iluminação, foco, ângulo e até dígitos vizinhos. Se
fossem divididos por recorte, o modelo veria no treino quase a mesma imagem que avalia no teste, e as métricas
ficariam otimistas (vazamento). Dividir por imagem também é necessário para calcular `acc_reading`. Todos os grupos
usam o mesmo `splits.csv` para que os números sejam comparáveis entre grupos: com só 187 imagens de teste, uma
divisão diferente mudaria os resultados mais do que muitas das diferenças entre modelos.

**3.2** Quantidade de faixas corrigidas: **120** (105 visores verticais girados 90° + 15 faixas horizontais de
cabeça para baixo giradas 180°). Em 880 faixas a orientação foi decidida pelos decimais vermelhos e em 363 pelo
classificador de orientação. Tipos de recorte errado e quantidades:
- **Orientação residual:** 13 faixas provavelmente ainda de cabeça para baixo (8 no treino, 3 na validação, 2 no
  teste). Todas foram decididas pelo classificador (sem vermelho), 10 delas em visores verticais. No teste elas
  respondem por 13 dos 60 erros da ResNet18 (21,7 %).
- **Desalinhamento / polígono frouxo:** faixas em que a grade de 8 partes não coincide com os tambores (margens,
  vírgula, tambor cortado na borda). No teste, as faixas com ≥ 4 dígitos errados (sem ser orientação) geram 14
  erros (23,3 %). Também encontramos um caso de perspectiva extrema (`id_419`, visor quase de perfil), em que os
  dígitos ficam girados 90° dentro da faixa.
- **Tambor em transição:** o dígito está rolando e mostra metade de dois números. 9 erros (15 %) são do tipo
  d → d±1, concentrados na última posição, que gira continuamente.
- **Outros** (borrão, reflexo, baixa iluminação, confusão de forma): 24 erros (40 %).

**3.3** A divisão em larguras iguais supõe que o polígono começa e termina exatamente nos tambores e que os 8 têm a
mesma largura e espaçamento. Na prática, o visor tem uma separação (vírgula/moldura) entre a parte inteira e a
decimal, margens diferentes à esquerda e à direita, e o `minAreaRect` de polígonos com 6–8 vértices inclui a
saliência da janela decimal. Por isso os recortes ficam deslocados e um dígito pode ser cortado entre dois recortes.
O efeito é maior na posição 7 (borda direita), a pior para todos os modelos (ResNet18: 0,914 contra ≥ 0,946 nas
demais). Correções: (i) localizar os tambores pelo perfil de projeção vertical (vales de intensidade entre dígitos)
ou por componentes conexas; (ii) usar os vértices do próprio polígono (a "quebra" entre inteiros e decimais) para
posicionar as fronteiras; (iii) recortar com uma pequena margem/sobreposição; ou (iv) evitar a divisão com um
modelo de sequência (CRNN + CTC) que lê a faixa inteira.

**4.1** Os pesos pré-treinados foram aprendidos com entradas normalizadas pela média e desvio do ImageNet. Os
filtros da primeira camada e as estatísticas das BatchNorms esperam essa escala. Com outra normalização, as
ativações saem da faixa em que as features fazem sentido e o *transfer learning* começa de um ponto pior. Usamos a
mesma normalização na `smallcnn` para manter o pré-processamento idêntico entre os modelos.

**4.2** smallcnn 0,93 M · ResNet18 11,18 M · AlexNet 57,04 M · VGG16-BN 134,31 M parâmetros. Na **AlexNet**, 95,7 %
(54,6 M) estão nas camadas totalmente conectadas do `classifier`, contra só 2,5 M nas convoluções. Na **VGG16-BN**,
89,0 % (119,6 M) estão no `classifier`, com 14,7 M nas convoluções. A primeira FC (`25088 → 4096`, ~103 M) sozinha
domina. A ResNet18 usa *global average pooling* e uma única FC de 5 k parâmetros, então 99,95 % estão nas
convoluções. Isso se reflete no checkpoint: 512 MB (VGG) e 218 MB (AlexNet) contra 43 MB (ResNet).

**4.3** Treinar só a cabeça usa o backbone como extrator de features fixo: é barato e não corre risco de destruir o
que foi aprendido, mas as features do ImageNet não são específicas para dígitos de hidrômetro. O *fine-tuning*
completo adapta todas as camadas e chega mais longe. Congelar o backbone na 1ª época evita que a cabeça recém
inicializada (aleatória) gere gradientes grandes e ruidosos que estragariam as features pré-treinadas. Primeiro a
cabeça aprende a usar as features; depois o resto é ajustado com LR menor. Nos nossos números, com o backbone
congelado a validação chegou a 0,515 / 0,526 / 0,362 de acurácia (AlexNet / VGG / ResNet). Uma época depois de
descongelar, foi a 0,806 / 0,826 / 0,864, ou seja, o ganho vem do *fine-tuning*.

**4.4** Dígitos não são simétricos: o flip horizontal produz formas que nunca aparecem no visor (2 e 5 espelhados,
7 invertido) e o vertical transforma 6 em 9. Além disso, a orientação faz parte do problema: corrigimos as faixas
justamente para que os dígitos apareçam sempre em pé. Um flip ensinaria o modelo a aceitar entradas impossíveis e
criaria rótulos errados (6 ↔ 9).

**5.1** A **ResNet18** (F1-macro 0,9536), seguida de VGG16-BN (0,9283), AlexNet (0,8968) e smallcnn (0,7522). O
ranking por acurácia é o mesmo, mas o F1-macro é sempre menor que a acurácia. O 0 é 34,4 % do teste e é a classe
mais fácil (recall 0,95–0,98). A acurácia pondera pelo suporte e é puxada por ele, enquanto o F1-macro dá o mesmo peso
às 10 classes e expõe as classes raras (8 e 9 na smallcnn, com F1 0,69 e 0,64). Por isso a diferença é maior na
smallcnn (0,801 × 0,752) e menor na ResNet18 (0,960 × 0,954).

**5.2** Somando os quatro modelos: 1→0 (35), 6→0 (23), 8→6 (18), 0→1 (17), 8→0 (17), 4→7 (16), 8→9 (16), 2→0 (16).
Há três causas. (i) **Viés para a classe majoritária:** recortes ambíguos (tambor rolando, borrão, recorte deslocado
que mostra a moldura) caem no 0. (ii) **Semelhança de forma:** 0, 6, 8 e 9 são laços fechados que diferem por um traço,
e 1, 4 e 7 têm hastes verticais parecidas. (iii) **Tambor em transição:** metade do 8 em cima e do 9 embaixo. A
smallcnn concentra a maior parte dessas confusões (por exemplo, 17 dos 23 casos 6→0).

**5.3** Uma leitura só está certa se os **8** dígitos estiverem certos. Se os erros fossem independentes, esperaríamos
`acc_digit^8`: 0,169 / 0,500 / 0,595 / 0,721 (smallcnn / AlexNet / VGG / ResNet). Os valores observados são
maiores (0,294 / 0,626 / 0,711 / 0,845) porque os erros se concentram nas mesmas imagens. Na ResNet18, 158 das 187
imagens de teste têm 0 erros, 18 têm 1 erro e só 11 concentram 2 a 7 erros (faixas de cabeça para baixo ou
desalinhadas). Mesmo assim, a exigência conjunta faz `acc_reading` cair muito mais que `acc_digit`.

**5.4** A **ResNet18**. Ela tem o melhor desempenho (F1 0,954, 84,5 % das leituras corretas) e é pequena: 11,2 M
parâmetros e checkpoint de 42,7 MB, cerca de 11 MB se quantizado para int8. Em CPU leva 9,4 ms por dígito, cerca de
75 ms para uma leitura de 8 dígitos (ou um único batch de 8). A VGG16-BN é pior (F1 0,928) e muito mais cara (512 MB,
40,5 ms/dígito em CPU). A AlexNet tem latência parecida (9,9 ms), mas é 5× maior (218 MB) e erra mais. A smallcnn é
a mais leve (3,6 MB, 2,4 ms), mas acerta só 29 % das leituras, o que a torna inútil para o usuário. Em um celular
real ainda valeria considerar a MobileNet, mas entre os modelos permitidos a ResNet18 é o melhor custo × benefício.

**5.5** Conexões residuais, BatchNorm em todas as camadas e *global average pooling* no lugar das FCs gigantes.
Com 12× menos parâmetros que a VGG16-BN (11,2 M × 134,3 M), a ResNet18 tem F1 2,5 p.p. maior, `acc_reading` 13,4 p.p.
maior (0,845 × 0,711), treina 4,2× mais rápido por época (6,9 s × 29,2 s), tem checkpoint 12× menor e é 4,3× mais
rápida em CPU. As conexões residuais facilitam o fluxo de gradiente no *fine-tuning*, e a ausência das FCs de 120 M
parâmetros reduz o sobreajuste com apenas 6960 recortes de treino. Na GPU com batch 1 as latências ficam próximas
(3,0 × 3,3 ms), porque nesse regime o custo é dominado pelo número de camadas sequenciais e não pelos FLOPs.

**5.6** Não. A smallcnn ficou com F1 0,752 (contra 0,897–0,954) e só 29,4 % das leituras corretas. As curvas
mostram que ela ainda estava aprendendo: a melhor época foi a última (10), o F1 de validação subia a cada época e a
acurácia de treino (0,767) ficou **abaixo** da de validação (0,810), um sinal de *underfitting* agravado pela
*augmentation* e pelo dropout. Isso indica que (i) com poucos dados (6960 recortes) e poucas épocas, as features
pré-treinadas no ImageNet dão uma vantagem enorme, e (ii) o protocolo de 10 épocas é curto para uma rede treinada do
zero. Com mais épocas ela provavelmente encostaria na AlexNet, sendo 60× menor e 4× mais rápida em CPU.

## Limitações

- **Erro de dados × erro de modelo:** dos 60 dígitos errados pela ResNet18 no teste, cerca de 27 (45 %) vêm de recortes
  ruins (13 de faixas de cabeça para baixo + 14 de faixas desalinhadas). Outros 9 (15 %) são tambores em transição,
  em que o próprio rótulo é ambíguo. Só cerca de 40 % parecem erros genuínos do modelo (borrão, reflexo, forma).
  Melhorar o pipeline de recorte provavelmente rende mais que trocar de arquitetura.
- **Orientação:** a heurística dos decimais vermelhos falha quando o visor não tem vermelho (363 faixas). O
  classificador de orientação treinado com pseudo-rótulos ainda deixa cerca de 13 faixas invertidas (1 %), quase
  todas verticais.
- **Divisão em larguras iguais** (Questão 3.3): a principal fonte de desalinhamento, sobretudo na posição 7.
- **Rótulos:** a leitura vem de anotação humana; tambores entre dois números têm rótulo ambíguo e o zero final dos
  decimais foi inferido (o float o perde).
- **Avaliação:** o teste tem só 187 imagens (1496 recortes), então diferenças de ~1 p.p. em `acc_digit` estão dentro do
  ruído. Rodamos uma única seed; os tempos dependem da GPU (RTX 5060 Ti) e não incluem o pré-processamento.
- **O que mudaríamos:** segmentar os tambores pelo perfil de projeção, usar o rótulo de orientação por consistência
  entre modelos, treinar a smallcnn por mais épocas e testar um modelo de sequência (CRNN + CTC) sobre a faixa
  inteira.

## Declaração de uso de IA

Preenchida conforme o *Template de Declaração de Uso de IA* da disciplina.

### Identificação da atividade

| Campo | Valor |
|---|---|
| Disciplina | Visão Computacional — VC001 |
| Data | 13 / 10 / 2026 |
| Alunos(as) e matrícula | Helder Barros — [matrícula] · Felipe Salustiano — [matrícula] · Lucas Mendes — [matrícula] · Lucas Rafael — [matrícula] · José Kevin — [matrícula] |
| Atividade | Prática II — Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet |

### Seção A — Uso de LLM / assistente de IA

| Campo | Valor |
|---|---|
| Utilizou IA? | ☒ Sim ☐ Não |
| Modelo(s) usado(s) | Claude Opus 5.5 (`claude-opus-5-5`, Anthropic), via extensão Claude Code para VS Code |
| Prompt(s) utilizado(s) | Ver lista abaixo (prompts exatos, na ordem em que foram enviados) |
| Resposta gerada | Ver descrição abaixo |
| Adaptações realizadas | Ver descrição abaixo |

**Prompts utilizados (texto exato):**

1. `@vc-20262-p2/README.md execute exatamente oq tem nesse readme. Crie uma nova pasta para o projeto dentro da pasta "VC", separada da pasta "vc-20262-p2". utilize os dados de @vc-20262-p2/data/`
2. `oq está faltando?`
3. `n seria interessante se eu mesmo rodar o notebook?`
4. `como eu sei como está o andamento da sua?`
5. `pode me dizer como faço para acompanhar?`
6. (colado do terminal) `while ($true) { Clear-Host; ... }` — `'while' não é reconhecido como um comando interno ou externo, um programa operável ou um arquivo em lotes.`
7. `Quais são os arquivos que eu tenho que colocar no repositório do git?`
8. `a maneira como o projeto está estruturado, é a melhor? o history na raiz do projeto, o notebbok na raiz também. acho q está desorganizado. poderia reorganizar? se reorganizar alguma coisa vai mudar?`
9. `oq preciso fazer agr?`
10. `1) utilize esse template fornecido pelo professor para preencher a declaração do uso de IA 2) irei revisar e reescrever as respostas. 3) nao irei fazer isso para não alterar os resultados 4) irei criar o repositório agora, mas nao quero colocar o readme agora, como faço?` (com o PDF deste template anexado)

**Resposta gerada (incorporada à entrega):**

- Todo o código do notebook `pratica_watermeters.ipynb`: aquisição e
  verificação de integridade, análise exploratória, construção do rótulo de 8 dígitos, retificação da faixa 384×64,
  correção de orientação (heurística dos decimais vermelhos + classificador auxiliar de orientação), recortes 48×64,
  `digits/index.csv`, treino dos quatro modelos com o protocolo do roteiro, avaliação, figuras, galerias e diagnóstico
  dos tipos de erro.
- O script `tools/clear_image_outputs.py`, o `requirements.txt` e o `.gitignore`.
- A configuração do ambiente (venv com PyTorch CUDA 12.8) e a execução completa do notebook, que gerou
  `results.csv`, `history_*.csv`, `figures/` e `reports/`.
- Um **rascunho** deste README: reprodução, protocolo, tabela de resultados, respostas às questões e limitações.
- Orientações sobre como acompanhar a execução, quais arquivos versionar e a estrutura do repositório.

**Adaptações realizadas (validação e revisão pelo grupo):**

- A execução foi validada pelas verificações obrigatórias do roteiro, que o próprio notebook testa com `assert`:
  1243 imagens válidas, 1 descartada, 9944 recortes e nenhuma imagem em mais de um split. O `splits.csv` oficial
  foi usado sem alteração.
- Os números do README foram conferidos com o `results.csv` versionado.
- A estrutura do projeto foi mantida como pede o roteiro (entregáveis na raiz), após discutirmos com o assistente se
  valia reorganizar. O grupo decidiu não executar o notebook novamente para não alterar os resultados.
- As respostas às questões e a seção de limitações foram revisadas e reescritas pelo grupo a partir dos resultados
  obtidos: **[descrever aqui o que o grupo alterou, corrigiu ou acrescentou]**.

### Seção B — Uso de skill (Claude Code / SKILL.md)

| Campo | Valor |
|---|---|
| Utilizou skill? | ☒ Sim ☐ Não |
| Nome da skill | `dataviz` (skill embutida do Claude Code) |
| Conteúdo do SKILL.md | Anexado em [`docs/ia/SKILL_dataviz.md`](docs/ia/SKILL_dataviz.md). Foi usada só como guia de estilo das figuras (paleta categórica fixa por modelo, linhas finas, legenda e rótulos diretos). |
| Versão / hash | Versão empacotada no Claude Code 2.1.296; carregada em 13/10/2026 (o arquivo não fica salvo em disco, por isso não há hash) |

### Seção C — Uso de SDD (System Design Document)

| Campo | Valor |
|---|---|
| Utilizou SDD? | ☐ Sim ☒ Não |
| Arquivo SDD | — |
| Partes geradas por IA | — |

### Declaração de responsabilidade

Declaramos que:

- As informações prestadas nesta declaração são verdadeiras e completas.
- Compreendemos e somos responsáveis por todo o conteúdo entregue, incluindo as partes geradas com auxílio de IA.
- Estamos cientes de que a omissão do uso de IA configura conduta academicamente desonesta.
