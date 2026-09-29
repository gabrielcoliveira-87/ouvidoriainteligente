"""
Ouvidoria Inteligente - Buscador semantico de manifestacoes (Entrega 4)

Como rodar:
    pip install streamlit pandas numpy scikit-learn matplotlib sentence-transformers langchain-text-splitters
    streamlit run app_ouvidoria.py

Abas: Busca Semantica | Base Completa | Espaco Vetorial | Chunking

Sobre os dados:
  - Por padrao o app carrega DADOS_EXEMPLO (40 manifestacoes ficticias escritas so para
    testar o app; NAO sao o dataset oficial da disciplina).
  - Para usar o dataset oficial, envie o CSV pela sidebar (colunas: id, categoria, texto).

Sobre os modelos:
  - Se o sentence-transformers ou o modelo escolhido nao estiver disponivel (ex: sem
    internet na 1a execucao), o app cai automaticamente num FALLBACK esparso
    (TF-IDF de n-gramas de caracteres) e avisa na sidebar. O fallback NAO captura
    sinonimos como um embedding denso; serve so para o app continuar funcionando.
"""

import html
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

try:
    from sentence_transformers import SentenceTransformer
    TEM_SBERT = True
except ImportError:
    TEM_SBERT = False

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    TEM_LANGCHAIN = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        TEM_LANGCHAIN = True
    except ImportError:
        TEM_LANGCHAIN = False


# =====================================================================
# Dados
# =====================================================================

CATEGORIAS = ["infraestrutura", "saúde", "segurança", "educação", "meio ambiente"]

# 40 manifestacoes FICTICIAS (id, categoria, texto) so para testar o app.
# Tem 3 pares de duplicatas semanticas (M003/M017, M008/M022, M012/M035 = 15%) e
# 5 textos longos (>500 caracteres, para a aba de chunking): M005, M014, M026, M029, M038.
DADOS_EXEMPLO = [
    ("M001", "infraestrutura", "Falta calçamento na rua dos Coqueiros e a lama toma conta quando chove, dificultando a passagem dos moradores."),
    ("M002", "saúde", "A UPA do bairro está sem ambulância disponível há duas semanas e os pacientes graves esperam transferência por horas."),
    ("M003", "infraestrutura", "Existe um buraco enorme na Av. Brasil, na altura do mercado, e os carros desviam de forma perigosa para não quebrar o eixo."),
    ("M004", "educação", "A escola municipal Paulo Freire está sem professor de matemática desde o início do semestre e os alunos perdem conteúdo."),
    ("M005", "infraestrutura", "A ponte de madeira que liga o bairro Alto da Boa Vista ao centro da cidade está em situação crítica. Várias tábuas estão soltas ou apodrecidas, o corrimão caiu há dois meses e ninguém da prefeitura veio consertar. Diariamente crianças atravessam a ponte para ir à escola e motoqueiros passam em alta velocidade, o que já causou duas quedas na última semana. Quando chove, a estrutura fica escorregadia e balança bastante. Já fizemos abaixo-assinado e ligamos várias vezes para a secretaria de obras, mas não recebemos nenhuma resposta. Pedimos providências urgentes antes que aconteça uma tragédia."),
    ("M006", "segurança", "Assaltos frequentes na parada de ônibus da rua XV à noite, sem policiamento nem iluminação adequada."),
    ("M007", "meio ambiente", "Moradores estão queimando lixo em terreno baldio ao lado da escola, e a fumaça causa problemas respiratórios nas crianças."),
    ("M008", "saúde", "O posto de saúde do bairro Jardim Esperança está sem médico há um mês e as pessoas voltam para casa sem atendimento."),
    ("M009", "educação", "Faltam merenda e material de limpeza na creche municipal, e as crianças ficam com fome durante a tarde inteira."),
    ("M010", "segurança", "Carros trafegam em alta velocidade em frente ao colégio e não há faixa de pedestres nem lombada para proteger os alunos."),
    ("M011", "meio ambiente", "O esgoto a céu aberto na rua das Palmeiras escorre até o córrego e o mau cheiro atinge toda a vizinhança."),
    ("M012", "segurança", "Há um grupo usando drogas todas as noites na praça central, o que intimida as famílias e ameaça a segurança dos frequentadores."),
    ("M013", "educação", "O transporte escolar rural está atrasando quase todos os dias e os estudantes chegam quando a aula já começou."),
    ("M014", "saúde", "Fui ao hospital municipal com minha mãe, que tem 78 anos e sofre de insuficiência cardíaca, e ela passou mais de nove horas na recepção sem ser chamada para a triagem. Não havia cadeiras suficientes, o ar-condicionado estava quebrado e ninguém explicava quanto tempo ainda demoraria. Quando finalmente foi atendida, o médico disse que faltava o aparelho de eletrocardiograma e que ela precisaria fazer o exame em outra unidade, a quinze quilômetros de distância. Voltamos para casa sem diagnóstico e sem medicação. Peço que a ouvidoria investigue a falta de equipamentos e de profissionais nesse hospital."),
    ("M015", "infraestrutura", "A rede de água da rua Santos Dumont fica sem abastecimento durante o dia inteiro, e a companhia não dá previsão de reparo."),
    ("M016", "meio ambiente", "Caminhões descartam entulho de obras em uma área de preservação perto do rio, destruindo a vegetação nativa."),
    ("M017", "infraestrutura", "O asfalto da avenida principal está todo esburacado e é praticamente impossível dirigir sem danificar o carro."),
    ("M018", "saúde", "Não há vacina da gripe disponível na unidade básica, e os idosos foram orientados a voltar semanas depois sem data definida."),
    ("M019", "educação", "O teto da sala de aula do 5º ano tem infiltração e, quando chove, os alunos precisam ser realocados no pátio."),
    ("M020", "segurança", "As câmeras de monitoramento do bairro estão quebradas desde o ano passado e os furtos de bicicletas aumentaram muito."),
    ("M021", "meio ambiente", "Árvores centenárias da praça foram podadas de forma drástica, sem laudo técnico e sem aviso à comunidade."),
    ("M022", "saúde", "Está faltando atendimento no PSF do nosso bairro, os moradores chegam cedo e não conseguem consulta com nenhum profissional."),
    ("M023", "infraestrutura", "Os bueiros da rua do Comércio estão entupidos e a cada chuva forte a água invade as lojas e as casas vizinhas."),
    ("M024", "educação", "A biblioteca da escola estadual está fechada por falta de funcionário, e os estudantes não têm onde pesquisar."),
    ("M025", "saúde", "Faltam medicamentos para pressão alta e diabetes na farmácia do posto, e os pacientes precisam comprar por conta própria."),
    ("M026", "segurança", "Moro na rua Dom Pedro há dez anos e nunca vi a situação tão grave. Desde março, um grupo de motoqueiros faz rachas na madrugada, com barulho ensurdecedor e disparos de escapamento que parecem tiros. Na semana passada, um vizinho foi ameaçado ao pedir que parassem e teve o portão danificado. Já acionamos a polícia militar diversas vezes, mas as viaturas nunca chegam a tempo, e a última vez em que ligaram informaram que não havia efetivo disponível. Os moradores estão com medo de retaliação e várias famílias pensam em se mudar. Solicitamos rondas regulares e uma solução definitiva."),
    ("M027", "meio ambiente", "Uma fábrica próxima ao rio despeja resíduos de cor escura na água, e peixes mortos apareceram na margem nesta semana."),
    ("M028", "infraestrutura", "O ponto de ônibus da avenida Central não tem cobertura nem assento e os idosos esperam em pé debaixo de sol forte."),
    ("M029", "educação", "Meu filho estuda no 3º ano do ensino médio em uma escola estadual da zona norte e, desde o início do ano, os alunos estão sem aulas de física e de química porque os professores foram remanejados e ninguém foi contratado para substituí-los. A direção diz que já pediu profissionais à secretaria de educação, mas até agora nada foi feito. Os estudantes vão fazer o vestibular no fim do ano e estão muito prejudicados, pois as famílias mais pobres não podem pagar cursinho ou aulas particulares. Pedimos que a secretaria contrate os professores com urgência e garanta o conteúdo perdido por meio de aulas de reforço."),
    ("M030", "segurança", "Moradores relatam tiros à noite perto do campo de futebol e temem sair de casa depois das oito horas."),
    ("M031", "infraestrutura", "Tem uma lâmpada queimada na praça da matriz há semanas e o local ficou totalmente escuro à noite."),
    ("M032", "saúde", "A fila para marcar exame de ultrassom no posto passa de seis meses e as gestantes estão sem acompanhamento adequado."),
    ("M033", "educação", "Não há acessibilidade na escola: falta rampa e banheiro adaptado para o aluno cadeirante da turma do 8º ano."),
    ("M034", "meio ambiente", "A coleta seletiva não passa no bairro há meses e o lixo reciclável está se acumulando nas calçadas."),
    ("M035", "segurança", "Pessoas consomem entorpecentes toda noite na praça do centro, deixando as famílias com medo de frequentar o espaço."),
    ("M036", "infraestrutura", "O muro de contenção da encosta na rua Alegre apresenta rachaduras grandes e há risco de desabamento."),
    ("M037", "saúde", "As ambulâncias do SAMU demoram mais de uma hora para chegar ao distrito rural, o que já colocou vidas em risco."),
    ("M038", "meio ambiente", "Há mais de três meses um lixão clandestino se formou em um terreno vazio na entrada do bairro Novo Horizonte. No início eram apenas alguns sacos de lixo, mas hoje há colchões, restos de construção, pneus e até animais mortos. O chorume escorre para uma nascente que abastece várias famílias da comunidade rural, e o mau cheiro é insuportável, principalmente nos dias quentes. Aumentaram os casos de dengue e de doenças de pele entre as crianças. Já pedimos à limpeza urbana que retire os resíduos e fiscalize quem despeja, mas ninguém compareceu. Pedimos a limpeza imediata do local e a instalação de câmeras."),
    ("M039", "segurança", "Os postes sem luz na rua da escola deixam o caminho perigoso para os alunos do turno da noite."),
    ("M040", "educação", "Faltam computadores funcionando no laboratório de informática e os alunos não conseguem realizar as aulas práticas."),
]

# alternativas de nome de coluna aceitas no CSV oficial
COLUNAS_ACEITAS = {
    "id": ["id", "codigo", "código", "protocolo", "id_manifestacao"],
    "categoria": ["categoria", "category", "tema"],
    "texto": ["texto", "descricao", "descrição", "manifestacao", "manifestação"],
}


def base_exemplo():
    return pd.DataFrame(DADOS_EXEMPLO, columns=["id", "categoria", "texto"])


def ler_csv(arquivo):
    """Le o CSV enviado e padroniza as colunas para id / categoria / texto."""
    df = pd.read_csv(arquivo, sep=None, engine="python")  # sep=None: detecta , ou ;
    minusculas = {c.lower().strip(): c for c in df.columns}
    renomear = {}
    for padrao, opcoes in COLUNAS_ACEITAS.items():
        achou = next((minusculas[o] for o in opcoes if o in minusculas), None)
        if achou is None:
            raise ValueError(f"Não encontrei a coluna '{padrao}'. Colunas do arquivo: {list(df.columns)}")
        renomear[achou] = padrao
    df = df.rename(columns=renomear)[["id", "categoria", "texto"]].dropna()
    return df.reset_index(drop=True)


# =====================================================================
# Embeddings (modelo denso ou fallback esparso)
# =====================================================================

FALLBACK = "TF-IDF de caracteres (fallback, sem modelo)"
MODELOS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "BAAI/bge-small-pt-v1.5",
    "paraphrase-multilingual-mpnet-base-v2",
    FALLBACK,
]


@st.cache_resource
def carregar_modelo(nome):
    """Carrega o SentenceTransformer uma unica vez. Devolve None se nao der."""
    if nome == FALLBACK or not TEM_SBERT:
        return None
    try:
        return SentenceTransformer(nome)
    except Exception:
        return None


@st.cache_resource
def vetorizador_tfidf(referencia):
    """Ajusta o TF-IDF (n-gramas de caracteres) no corpus de referencia."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                          strip_accents="unicode", lowercase=True)
    vec.fit(list(referencia))
    return vec


@st.cache_data(show_spinner="Gerando embeddings...")
def codificar(nome_modelo, textos, referencia):
    """
    textos e referencia sao TUPLAS (para o cache_data conseguir fazer hash).
    Devolve (matriz_normalizada, modo) onde modo = 'modelo' ou 'fallback'.
    """
    modelo = carregar_modelo(nome_modelo)
    if modelo is not None:
        emb = modelo.encode(list(textos), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=float), "modelo"
    vec = vetorizador_tfidf(referencia)
    emb = vec.transform(list(textos)).toarray()  # ja vem normalizado (L2)
    return emb, "fallback"


def matriz_similaridade(emb):
    """Vetores ja normalizados -> cosseno = produto interno."""
    return emb @ emb.T


# =====================================================================
# Duplicatas (Entrega 2)
# =====================================================================

def pares_acima_do_limiar(sim, limiar):
    """Todos os pares (i, j), i < j, com similaridade >= limiar (ordenado do maior p/ menor)."""
    linhas, colunas = np.triu_indices_from(sim, k=1)
    pares = [(int(i), int(j), float(sim[i, j])) for i, j in zip(linhas, colunas) if sim[i, j] >= limiar]
    return sorted(pares, key=lambda p: -p[2])


def detectar_duplicatas(textos, limiar=0.85, modelo=MODELOS[0]):
    """Recebe a lista de manifestacoes e devolve os pares (i, j, similaridade) acima do limiar."""
    ref = tuple(textos)
    emb, _ = codificar(modelo, ref, ref)
    return pares_acima_do_limiar(matriz_similaridade(emb), limiar)


# =====================================================================
# Chunking (Entrega 3)
# =====================================================================

def _recursivo_proprio(texto, tamanho, overlap, separadores=("\n\n", "\n", " ", "")):
    """Versao propria do RecursiveCharacterTextSplitter (usada so se o langchain nao estiver instalado)."""

    def juntar(pedacos, sep):
        docs, atual, total = [], [], 0
        for p in pedacos:
            extra = len(sep) if atual else 0
            if total + len(p) + extra > tamanho and atual:
                doc = sep.join(atual).strip()
                if doc:
                    docs.append(doc)
                # descarta do inicio ate sobrar so o overlap
                while atual and (total > overlap or
                                 (total + len(p) + (len(sep) if atual else 0) > tamanho and total > 0)):
                    total -= len(atual[0]) + (len(sep) if len(atual) > 1 else 0)
                    atual = atual[1:]
            atual.append(p)
            total += len(p) + (len(sep) if len(atual) > 1 else 0)
        doc = sep.join(atual).strip()
        if doc:
            docs.append(doc)
        return docs

    def dividir(txt, seps):
        sep, proximos = seps[-1], []
        for i, s in enumerate(seps):
            if s == "":
                sep = s
                break
            if s in txt:
                sep, proximos = s, list(seps[i + 1:])
                break
        partes = txt.split(sep) if sep else list(txt)
        finais, bons = [], []
        for parte in partes:
            if len(parte) < tamanho:
                bons.append(parte)
            else:
                if bons:
                    finais += juntar(bons, sep)
                    bons = []
                finais += [parte] if not proximos else dividir(parte, proximos)
        if bons:
            finais += juntar(bons, sep)
        return finais

    return dividir(texto, list(separadores))


def dividir_texto(texto, estrategia, tamanho, overlap):
    """Quebra o texto em chunks segundo a estrategia escolhida na interface."""
    if estrategia.startswith("Recursive"):
        if TEM_LANGCHAIN:
            splitter = RecursiveCharacterTextSplitter(chunk_size=tamanho, chunk_overlap=overlap)
            return splitter.split_text(texto)
        return _recursivo_proprio(texto, tamanho, overlap)

    if estrategia.startswith("Tamanho fixo"):
        passo = max(1, tamanho - overlap)
        return [texto[i:i + tamanho].strip() for i in range(0, len(texto), passo) if texto[i:i + tamanho].strip()]

    # "Por sentenças": junta sentencas ate estourar o tamanho; overlap>0 repete a ultima sentenca
    sentencas = [s.strip() for s in re.split(r"(?<=[.!?])\s+", texto) if s.strip()]
    chunks, atual = [], []
    for s in sentencas:
        if atual and len(" ".join(atual + [s])) > tamanho:
            chunks.append(" ".join(atual))
            atual = atual[-1:] if overlap > 0 else []
        atual.append(s)
    if atual:
        chunks.append(" ".join(atual))
    return chunks


# =====================================================================
# Visualizacoes
# =====================================================================

@st.cache_data
def reduzir_2d(emb, metodo):
    if metodo == "PCA":
        return PCA(n_components=2, random_state=0).fit_transform(emb)
    n = len(emb)
    perplexidade = max(2, min(30, (n - 1) // 3))  # perplexidade precisa ser menor que n
    return TSNE(n_components=2, perplexity=perplexidade, init="pca", metric="cosine",
                random_state=0).fit_transform(emb)


def figura_heatmap(sim, ids):
    fig, ax = plt.subplots(figsize=(8, 7))
    img = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=90, fontsize=5)
    ax.set_yticklabels(ids, fontsize=5)
    fig.colorbar(img, ax=ax, label="similaridade de cosseno")
    ax.set_title("Matriz de similaridade")
    fig.tight_layout()
    return fig


def figura_espaco(pontos, categorias, ids, metodo):
    fig, ax = plt.subplots(figsize=(8, 6))
    cores = plt.get_cmap("tab10")
    for k, cat in enumerate(sorted(set(categorias))):
        mask = np.array([c == cat for c in categorias])
        ax.scatter(pontos[mask, 0], pontos[mask, 1], color=cores(k), label=cat, s=45, alpha=0.85)
    for (x, y), i in zip(pontos, ids):
        ax.annotate(i, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"Espaço vetorial das manifestações ({metodo})")
    ax.set_xlabel("componente 1")
    ax.set_ylabel("componente 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def figura_chunks(pontos):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pontos[:, 0], pontos[:, 1], "-", color="lightgray", zorder=1)
    ax.scatter(pontos[:, 0], pontos[:, 1], c=range(len(pontos)), cmap="viridis", s=90, zorder=2)
    for k, (x, y) in enumerate(pontos, start=1):
        ax.annotate(str(k), (x, y), ha="center", va="center", fontsize=8, color="white", zorder=3)
    ax.set_title("Chunks no espaço 2D (PCA) — a linha segue a ordem do texto")
    fig.tight_layout()
    return fig


def cor_do_score(score):
    if score > 0.7:
        return "#d4edda", "#28a745"   # verde
    if score > 0.5:
        return "#fff3cd", "#ffc107"   # amarelo
    return "#e9ecef", "#adb5bd"       # cinza


# =====================================================================
# Interface
# =====================================================================

def main():
    st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="📣", layout="wide")
    st.title("📣 Ouvidoria Inteligente — busca semântica de manifestações")

    # ---------- Sidebar ----------
    st.sidebar.header("⚙️ Configurações")
    nome_modelo = st.sidebar.selectbox("Modelo de embedding", MODELOS)
    top_k = st.sidebar.slider("Top-k resultados", 1, 10, 5)
    arquivo = st.sidebar.file_uploader("CSV da base (id, categoria, texto)", type=["csv"])

    if arquivo is not None:
        try:
            df = ler_csv(arquivo)
            st.sidebar.success(f"Base carregada: {len(df)} manifestações.")
        except Exception as erro:
            st.sidebar.error(f"CSV inválido: {erro}. Usando a base de exemplo.")
            df = base_exemplo()
    else:
        df = base_exemplo()
        st.sidebar.info("Usando 40 manifestações FICTÍCIAS de exemplo (não é o dataset oficial).")

    textos = tuple(df["texto"])
    emb, modo = codificar(nome_modelo, textos, textos)
    if modo == "fallback":
        if nome_modelo != FALLBACK:
            st.sidebar.warning(f"Não consegui carregar '{nome_modelo}' (sentence-transformers ou internet "
                               "indisponível). Usando TF-IDF de caracteres como fallback.")
        else:
            st.sidebar.warning("Modo fallback esparso: não captura sinônimos como um embedding denso.")
    else:
        st.sidebar.success(f"Modelo carregado: {nome_modelo}")
    if not TEM_LANGCHAIN:
        st.sidebar.caption("langchain não instalado: usando splitter recursivo próprio (mesmo algoritmo).")

    aba_busca, aba_base, aba_espaco, aba_chunk = st.tabs(
        ["🔎 Busca Semântica", "📋 Base Completa", "🗺️ Espaço Vetorial", "✂️ Chunking"])

    # ---------- Aba 1: busca ----------
    with aba_busca:
        st.subheader("Descreva o problema com suas palavras")
        consulta = st.text_area("Descrição livre", value="asfalto cheio de buracos na rua",
                                height=90, key="consulta")
        st.caption("Cores: 🟩 score > 0.7 · 🟨 score > 0.5 · ⬜ demais")
        if consulta.strip():
            q, _ = codificar(nome_modelo, (consulta,), textos)
            scores = emb @ q[0]
            for pos in np.argsort(-scores)[:top_k]:
                fundo, borda = cor_do_score(scores[pos])
                st.markdown(
                    f"<div style='background:{fundo};border-left:6px solid {borda};padding:10px 14px;"
                    f"margin-bottom:8px;border-radius:6px;color:#222'>"
                    f"<b>{html.escape(str(df['id'][pos]))}</b> · {html.escape(str(df['categoria'][pos]))} · "
                    f"<b>score {scores[pos]:.3f}</b><br>{html.escape(df['texto'][pos])}</div>",
                    unsafe_allow_html=True)
        else:
            st.info("Digite uma descrição para buscar.")

    # ---------- Aba 2: base completa ----------
    with aba_base:
        st.subheader(f"Base completa ({len(df)} manifestações)")
        tabela = df.copy()
        tabela["caracteres"] = tabela["texto"].str.len()
        st.dataframe(tabela, hide_index=True, use_container_width=True)

        if st.button("Gerar matriz de similaridade"):
            st.session_state["mostrar_matriz"] = True
        if st.session_state.get("mostrar_matriz"):
            sim = matriz_similaridade(emb)
            fig = figura_heatmap(sim, list(df["id"]))
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("#### Possíveis duplicatas")
            dinamico = st.checkbox("Limiar dinâmico (percentil 90 das similaridades)")
            if dinamico:
                valores = sim[np.triu_indices_from(sim, k=1)]
                limiar = float(np.percentile(valores, 90))
                st.write(f"Limiar = percentil 90 = **{limiar:.3f}**")
            else:
                limiar = st.slider("Limiar fixo", 0.0, 1.0, 0.85, 0.01)
            pares = pares_acima_do_limiar(sim, limiar)
            if pares:
                st.dataframe(pd.DataFrame([{
                    "A": df["id"][i], "B": df["id"][j], "similaridade": round(s, 3),
                    "categoria A": df["categoria"][i], "categoria B": df["categoria"][j],
                    "texto A": df["texto"][i], "texto B": df["texto"][j]} for i, j, s in pares]),
                    hide_index=True, use_container_width=True)
            else:
                st.info("Nenhum par acima desse limiar.")

    # ---------- Aba 3: espaco vetorial ----------
    with aba_espaco:
        st.subheader("Espaço vetorial das manifestações")
        metodo = st.radio("Redução de dimensionalidade", ["PCA", "t-SNE"], horizontal=True)
        pontos = reduzir_2d(emb, "PCA" if metodo == "PCA" else "t-SNE")
        fig = figura_espaco(pontos, list(df["categoria"]), list(df["id"]), metodo)
        st.pyplot(fig)
        plt.close(fig)

        # metricas objetivas para comentar se os clusters coincidem com as categorias
        sim = matriz_similaridade(emb).copy()
        np.fill_diagonal(sim, -np.inf)
        vizinho = sim.argmax(axis=1)
        cats = df["categoria"].to_numpy()
        acerto = float((cats[vizinho] == cats).mean())
        try:
            silhueta = float(silhouette_score(emb, cats, metric="cosine"))
        except ValueError:
            silhueta = float("nan")
        c1, c2 = st.columns(2)
        c1.metric("Vizinho mais próximo é da mesma categoria", f"{acerto:.0%}")
        c2.metric("Silhueta por categoria (cosseno)", f"{silhueta:.2f}")
        if acerto >= 0.8:
            leitura = "Os clusters semânticos **coincidem bem** com as categorias oficiais."
        elif acerto >= 0.5:
            leitura = ("Os clusters coincidem **parcialmente** com as categorias: há temas que cruzam categorias "
                       "(ex.: iluminação pública pode ser infraestrutura ou segurança).")
        else:
            leitura = "Os clusters **não coincidem** com as categorias oficiais neste modelo."
        st.markdown(f"**Leitura automática:** {leitura} Silhueta próxima de 0 ou negativa indica categorias "
                    "sobrepostas no espaço vetorial.")
        st.caption("t-SNE preserva vizinhanças locais, mas distorce distâncias globais; PCA preserva a "
                   "variância global mas costuma juntar grupos em 2D.")

    # ---------- Aba 4: chunking ----------
    with aba_chunk:
        st.subheader("Chunking de manifestações longas")
        mais_longas = df.assign(n=df["texto"].str.len()).nlargest(5, "n")
        opcoes = ["(colar meu próprio texto)"] + [f"{r.id} ({r.n} caracteres)" for r in mais_longas.itertuples()]
        escolha = st.selectbox("Carregar uma das 5 manifestações mais longas", opcoes)
        inicial = "" if escolha == opcoes[0] else mais_longas["texto"].iloc[opcoes.index(escolha) - 1]
        texto_longo = st.text_area("Manifestação longa", value=inicial, height=160, key=f"longo_{escolha}")

        c1, c2, c3 = st.columns(3)
        estrategia = c1.selectbox("Estratégia", [
            "Recursive (LangChain)", "Tamanho fixo (caracteres)", "Por sentenças"])
        tamanho = c2.slider("chunk_size", 100, 800, 250, 10)
        overlap = c3.slider("chunk_overlap", 0, 300, 50, 10)
        if overlap >= tamanho:
            st.warning("O overlap precisa ser menor que o chunk_size; ajustei automaticamente.")
            overlap = tamanho - 1

        if texto_longo.strip():
            chunks = dividir_texto(texto_longo, estrategia, tamanho, overlap)
            st.write(f"**{len(chunks)} chunks** gerados "
                     f"(tamanho médio: {np.mean([len(c) for c in chunks]):.0f} caracteres).")
            emb_chunks, _ = codificar(nome_modelo, tuple(chunks), textos)
            emb_total, _ = codificar(nome_modelo, (texto_longo,), textos)

            tab = pd.DataFrame({
                "chunk": range(1, len(chunks) + 1),
                "caracteres": [len(c) for c in chunks],
                "sim. com texto completo": (emb_chunks @ emb_total[0]).round(3),
                "sim. com chunk anterior": [None] + list((emb_chunks[1:] * emb_chunks[:-1]).sum(axis=1).round(3)),
                "texto": chunks,
            })
            st.dataframe(tab, hide_index=True, use_container_width=True)
            st.caption("'Sim. com chunk anterior' mede a coesão entre chunks consecutivos: "
                       "aumente o overlap e veja o efeito.")

            with st.expander("Ver embeddings dos chunks"):
                st.write(f"Dimensão do vetor: {emb_chunks.shape[1]}")
                st.dataframe(pd.DataFrame(emb_chunks[:, :8].round(3),
                                          columns=[f"d{i}" for i in range(min(8, emb_chunks.shape[1]))],
                                          index=[f"chunk {i}" for i in range(1, len(chunks) + 1)]),
                             use_container_width=True)
                st.caption("Mostrando só as 8 primeiras dimensões de cada vetor.")

            if len(chunks) >= 3:
                fig = figura_chunks(PCA(n_components=2, random_state=0).fit_transform(emb_chunks))
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("São necessários pelo menos 3 chunks para o gráfico 2D. Diminua o chunk_size.")
        else:
            st.info("Cole um texto ou escolha uma manifestação da lista.")


if __name__ == "__main__":
    main()
