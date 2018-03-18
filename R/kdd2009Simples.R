#
# Desafio KDD 2009
#
# Roberto Stelling
# roberto@stelling.cc
#

# Diretório dos dados
path <- ".."
setwd(path)

# Leitura dos dados
dados <- read.table('data/orange_small_train.data', sep='\t', header=T, na.strings=c(''))

# Leitura dos rótulos dos dados
appetency <- read.table('data/orange_small_train_appetency.labels', sep='\t', header=F, col.names=c("appetency"))
churn <- read.table('data/orange_small_train_churn.labels', sep='\t', header=F, col.names=c("churn"))
upselling <- read.table('data/orange_small_train_upselling.labels', sep='\t', header=F, col.names=c("upselling"))

# Inclusão dos rótulos no conjunto de dados como colunas adicionais
# Com valores em booleanos:
# positivo (1): TRUE
# negativo (-1): FALSE
objetivo <- 1
dados$appetency <- appetency$appetency == objetivo
dados$churn <- churn$churn == objetivo
dados$upselling <- upselling$upselling == objetivo


# Cria lista de variáveis, excluindo as colunas de resultados (labels)
resultados <- c('appetency','churn','upselling')
variaveis <- setdiff(colnames(dados), resultados)
variaveisCategoricas <- variaveis[sapply(dados[,variaveis], class) %in% c('factor', 'character')]
variaveisNumericas <- variaveis[sapply(dados[,variaveis], class) %in% c('numeric', 'integer')]
variaveisVazias <- setdiff(variaveis, union(variaveisCategoricas, variaveisNumericas))

# Remove variaveis vazias e retira da lista 'variaveis'
dados <- dados[!names(dados) %in% c(variaveisVazias)]
variaveis <- setdiff(variaveis, variaveisVazias)

# Converte variaveis categoricas para numéricas
dados[variaveisCategoricas] <- sapply(dados[,variaveisCategoricas], as.numeric)

# Divide dados em treino e validação

# Número total de pontos de dados
numeroDados <- nrow(dados)

# Fixa semente para reproduzir resultados
set.seed(42)

# Define treino como 90% dos dados de treino e
# validação como os 10% restantes
tamanhoTreino <- floor(0.90 * numeroDados)

# Gera uma amostra aleatória com 90% dos índices
indicesTreino <- sample(seq_len(numeroDados), size=tamanhoTreino)

# Divide os dados nos dois subgrupos
# -indicesTreino são os 10% dos índices restantes
dadosTreino <- dados[indicesTreino, ]
dadosValidacao <- dados[-indicesTreino, ]

# Remove os data frames que não são mais necessários
rm(list=c('dados','churn','appetency','upselling'))

# Inclui bibliotecas
library('ggplot2')
library('WVPlots')
library('xgboost')
library('Metrics')

# Vetor para armazenar 
numResultados <- length(resultados)


# Ciclo de rodadas do XGBoost 20 a 300 de 2 em 2
listaNRounds <- seq(20, 300, by=2)

# Placar de resultado das simulações
placares <- matrix(1:(numResultados*length(listaNRounds)),
                   nrow=numResultados,
                   ncol=length(listaNRounds))
rownames(placares) <- resultados

# Parâmetros do XGBoost
params <- list(booster="gbtree", objective="binary:logistic",
               eta=0.05, gamma=0, max_depth=6, min_child_weight=1,
               subsample=1, colsample_bytree=1)

# Momento de início das simulações
inicio <- date()

for (resultado in resultados) {
  set.seed(4242)
  print(paste('Iniciando validação para: ', resultado))

  # Matriz de dados para XGBboost
  xgbDadosTreinamento <- xgb.DMatrix(as.matrix(dadosTreino[variaveis]),
                                               label=dadosTreino[,resultado])

  # Maior AUC e número de rodadas
  maximo <- -Inf
  melhor <- 0
  i <- 0
  treinamentoT <- dadosTreino[resultado]
  validacaoT <- dadosValidacao[resultado]

  print(paste('Iniciando boosting:', resultado))
  for(nrounds in listaNRounds) {
    i <- i + 1

    print(paste('  xgboost', resultado, nrounds,'rodadas'))

    modelo <- xgboost(data=xgbDadosTreinamento,
                            params=params,
                            verbose=0,
                            nrounds=nrounds)

    treinamentoT[['previsao']] <- as.numeric(predict(modelo, as.matrix(dadosTreino[variaveis])))
    validacaoT[['previsao']] <- as.numeric(predict(modelo, as.matrix(dadosValidacao[variaveis])))
 
    print(sprintf("  AUC Treinamento (%d): %5.4f", nrounds, auc(ifelse(treinamentoT[, resultado] == TRUE, 1, 0), treinamentoT['previsao'])))
  
    aucValidacao <- auc(ifelse(validacaoT[, resultado] == TRUE, 1, 0), validacaoT['previsao'])
    print(sprintf("  AUC Validação (%d): %5.4f", nrounds, aucValidacao))

    if (aucValidacao > maximo) {
        maximo <- aucValidacao
        melhor <- nrounds
    }
    placares[resultado, i] <- aucValidacao
  }
  print(paste(resultado, '  --> Melhor resultado', maximo,' - ', melhor, 'rodadas'))
  writeLines(paste('  Finalizando boosting', resultado, '\n'))
}

writeLines(paste('Validação finalizada', inicio, date()))

# Remove variáveis do ambiente para economizar memória
rm(list=c('treinamentoT', 'validacaoT', 'modelo', 'aucValidacao', 'melhor', 'maximo', 'nrounds', 'i',
          'resultado', 'indicesTreino', 'inicio', 'numeroDados',
          'path', 'tamanhoTreino', 'xgbDadosTreinamento'))

# Cria o dataframe para gerar os gráficos
performance <- data.frame(listaNRounds, placares[1,], placares[2,], placares[3,])
colnames(performance) <- c('rodadas', resultados)

# Gera os gráficos de performance das execuções do XGBoost
ggplot(performance, aes(rodadas)) + 
  geom_line(aes(y=appetency, colour="Appetency"), size=1) + 
  geom_line(aes(y=churn, colour="Churn"), size=1) +
  geom_line(aes(y=upselling, color="Upselling"), size=1) +
  labs( y='Performance', caption='Intervalos de 2',
        title='Performance do xgboost sobre dados do KDD 2009',
        subtitle='Validação', colour='Tipo Previsão')
ggsave("imagens/performance_simples.png", plot=last_plot(), device="png", scale=1)

ggplot(performance, aes(rodadas)) + 
  geom_line(aes(y=appetency, colour="Appetency"), size=1) + 
  geom_line(aes(y=churn, colour="Churn"), size=1) +
  geom_line(aes(y=upselling, color="Upselling"), size=1) +
  labs( y='Performance', caption='Intervalos de 2',
        title='Performance do xgboost sobre dados do KDD 2009',
        subtitle='Validação', colour='Tipo Previsão') +
  ylim(-0.01, 1.01)
ggsave("imagens/performance_0_1_simples.png", plot=last_plot(), device="png", scale=1)

ggplot(performance, aes(x=rodadas, y=appetency)) +
   geom_line(color="red", size=1) +
   labs( y='AUC - Appetency (Novos pacotes)',
         title='Appetency - AUC do xgboost de 20 a 300 rodadas')
ggsave("imagens/appetency_simples.png", plot=last_plot(), device="png", scale=1)

ggplot(performance, aes(x=rodadas, y=churn)) +
   geom_line(color="#009E73", size=1) +
   labs( y='AUC - Churn (Perda de clientes)',
         title='Churn - AUC do xgboost de 20 a 300 rodadas')

ggsave("imagens/churn__simples.png", plot=last_plot(), device="png", scale=1)

ggplot(performance, aes(x=rodadas, y=upselling)) +
   geom_line(color="blue", size=1) +
   labs( y='AUC - Up-selling (Upgrades de pacotes)',
         title='Up-selling - AUC do xgboost de 20 a 300 rodadas')
ggsave("imagens/upselling_simples.png", plot=last_plot(), device="png", scale=1)

# O resultado obtido na validação deve ser um limite máximo
# para a performance nos dados reais, já que é esperada uma
# performance menor nos dados de teste, que não participaram
# da modelagem.
# Simultâneamente serão geradas as previsões do modelo para
# o dataset de teste

numResultado <- 1
# Lê o arquivo de testes, para gerar as previsões do modelo
teste <- read.table('data/orange_small_test.data', sep='\t', header=T, na.strings=c(''))

teste[variaveisCategoricas] <- sapply(teste[variaveisCategoricas], as.numeric)

# Nome da coluna do dataframe de previsões
rodada <- 'previsao'

# Salva matrix de performances
write.table(performance, "output/performance_metricas_simples.csv", append=F, dec=".", sep=",", col.names=T, row.names=F)

for (resultado in resultados) {
  indice <- which.max(performance[,resultado,drop=T])
  rodadas <- performance$rodadas[indice]
  perfPrevisao <- performance[,resultado][indice]
  print(paste(resultado, perfPrevisao, rodadas, 'rodadas'))

  treinamentoT <- dadosTreino[resultado]
  validacaoT <- dadosValidacao[resultado]

  # Prepara dados para o xgb
  xgbDadosTreinamento <- xgb.DMatrix(as.matrix(dadosTreino[variaveis]),
                                               label=dadosTreino[,resultado])
  modelo <- xgboost(data=xgbDadosTreinamento,
                    params=params,
                    verbose=0,
                    nrounds=rodadas)

# Salva o modelo do  XGBoost
  xgb.save(modelo, paste('modelos/', resultado, '_simples.model', sep=''))

  treinamentoT[[rodada]] <- as.numeric(predict(modelo, as.matrix(dadosTreino[variaveis])))
  validacaoT[[rodada]] <- as.numeric(predict(modelo, as.matrix(dadosValidacao[variaveis])))

  # Salva as previsões do modelo no próprio dataframe de teste
  teste[resultado] <- as.numeric(predict(modelo, as.matrix(teste[variaveis])))
  teste[resultado] <- ifelse(teste[resultado] <= 0.5, -1, 1)
  # E gera os arquivos com as previsões
  write.table(teste[resultado], file=paste("output/orange_small_test_", resultado, "_simples.labels", sep=''),
              append=FALSE, quote=FALSE, row.names=FALSE, col.names=FALSE)

# Gera o plot de dupla densidade e ROC com as rotidas no WVPlots
  tituloTreinamento <- paste('Dados de treinamento:', rodadas)
  print(DoubleDensityPlot(treinamentoT, rodada, resultado, title=tituloTreinamento))
  ggsave(paste('imagens/ddTreinamento-', resultado, '_simples.png', sep=''), plot=last_plot(), scale=1)

  print(ROCPlot(treinamentoT, rodada, resultado, objetivo, title=tituloTreinamento))
  ggsave(paste('imagens/ROCPTreinamento-', resultado, '_simples.png', sep=''), plot=last_plot(), scale=1)

  print(sprintf("AUC Treinamento (%d): %5.4f", rodadas,
                auc(ifelse(treinamentoT[, resultado] == TRUE, 1, 0), treinamentoT[rodada])))

  tituloValidacao <- paste('Dados de validação:', rodadas)
  print(DoubleDensityPlot(validacaoT, rodada, resultado, title=tituloValidacao))
  ggsave(paste('imagens/ddValidacao-', resultado, '_simples.png', sep=''), plot=last_plot(), scale=1)

  print(ROCPlot(validacaoT, rodada, resultado, objetivo, title=tituloValidacao))
  ggsave(paste('imagens/ROCPValidacao-', resultado, '_simples.png', sep=''), plot=last_plot(), scale=1)
    
  aucValidacao <- auc(ifelse(validacaoT[, resultado] == TRUE, 1, 0), validacaoT[rodada])
  print(sprintf("AUC Validação (%d): %5.4f", rodadas, aucValidacao))
    
  numResultado <- numResultado + 1
}