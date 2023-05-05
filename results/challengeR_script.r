library(challengeR)
data_matrix = read.csv('merged-metric-log.csv')
challenge <- as.challenge(data_matrix, by='anatomy', algorithm='model',case='subject.id',value='DSC',
                          smallBetter=FALSE)
ranking <- challenge%>%testThenRank(alpha = 0.05, # significance level
                                    p.adjust.method = "none", # method for adjustment for
                                                              # multiple testing, see ?p.adjust
                                    na.treat = 0, # either "na.rm" to remove missing data,
                                                  # set missings to numeric value (e.g. 0)
                                                  # or specify a function, e.g. function(x) min(x)
                                    ties.method = "min" # a character string specifying
                                                        # how ties are treated, see ?base::rank
                                   )
set.seed(1)
ranking_bootstrapped <- ranking%>%bootstrap(nboot=1000)
meanRanks <- ranking%>%consensus(method = "euclidean") 
ranking_bootstrapped %>% report(consensus=meanRanks, title = 'Biplanar Xrayto3D  Benchmark',file='xray3D_benchmark',format='HTML')

# additional code to save svg
ordering_consensus=names(sort(t(ranking$matlist[[1]][,'rank',drop=F])['rank',]))
### Blob plots visualizing bootstrap results
stability(ranking,ordering=names(meanRanks))+theme(legend.position='none')
stabilityByTask(ranking_bootstrapped,ordering=names(meanRanks))+guides(color='none',fill='none')+theme(legend.position = 'none')

#blobplots
blob_pl=list()
for (subt in names(ranking_bootstrapped$bootsrappedRanks)){
  a=list(bootsrappedRanks=list(ranking_bootstrapped$bootsrappedRanks[[subt]]),
         matlist=list(ranking_bootstrapped$matlist[[subt]]))
  names(a$bootsrappedRanks)=names(a$matlist)=subt
  class(a)="bootstrap.list"
  r=ranking_bootstrapped$matlist[[subt]]

  blob_pl[[subt]]=stabilityByTask(a,
                             max_size =8,
                             ordering=rownames(r[order(r$rank),]),
                             size.ranks=.25*theme_get()$text$size,
                             size=8,
                             shape=4,
                             showLabelForSingleTask=showLabelForSingleTask) + 
    # scale_color_manual(values=cols) + #comment due to error
    guides(color = 'none')

}

if (length(ranking_bootstrapped$matlist)<=6 &nrow((ranking_bootstrapped$matlist[[1]]))<=10 ){
  ggpubr::ggarrange(plotlist = blob_pl)
} else {
  print(blob_pl)
}