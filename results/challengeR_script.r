library(challengeR)
data_matrix = read.csv('hip_metric_log.csv')
challenge <- as.challenge(data_matrix, algorithm='model',case='subject.id',value='DSC',
                          smallBetter=FALSE)
ranking=challenge%>%aggregateThenRank(FUN=mean,na.treat=0,method='min')
set.seed(1)
ranking_bootstrapped <- ranking%>%bootstrap(nboot=100)
ranking_bootstrapped %>% report(title = 'Hip Benchmark',file='hip_benchmark',
                                format='HTML')
