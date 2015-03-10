library('xlsx')
library('ggplot2')

incAxisLabelSpace = function(plot){
  # Input: ggplot object
  # Output: same ggplot object with increased space between axis
  #         and axis label
  new_plot = plot + theme(axis.title.x = element_text(vjust=-.1),
                          axis.title.y = element_text(vjust=1),
                          plot.title = element_text(vjust=.5))
  return(new_plot)
}

cutoff_vals <- read.xlsx2("GitHub/AdvancedMLProject/text_analysis/minimum_df_ngrams.xlsx", 
                        colClasses = c(rep("numeric", 5)), sheetIndex = 2, header=T)
feat_cutoffs <- cutoff_vals[,c('min_df', 'columns.features')]
feat_cutoffs$columns.features <- feat_cutoffs$columns.features*.001
cutoffs <- ggplot(feat_cutoffs, aes(y=columns.features, x=min_df)) 
cutoffs <- cutoffs + scale_x_log10("Minimum Freq (log10)")
cutoffs <- cutoffs + geom_line(color="blue") + ggtitle("N-gram Freq Cut-off vs. # of Features")
cutoffs <- cutoffs + ylab("# Ngram Features (000)")
cutoffs = incAxisLabelSpace(cutoffs)
cutoffs

ggsave(cutoffs, filename="Plots/ngramFeatCutoff.png", height=5, width=5, units="in")
