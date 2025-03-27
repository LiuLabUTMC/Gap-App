
#libraries
library(UCSCXenaTools)
library(dplyr)
library(tidyr)
library(tibble)
library(janitor)
library(data.table)
library(edgeR)
library(ggplot2)
library(fgsea)
library(ggeasy)
library(plotly)
library(clusterProfiler)
library(survival)
library(survminer)
library(magrittr)
library(stringr)
library(parallel)
library(foreach)
library(MASS)
library(doParallel)
library(readxl)
library(writexl)
library(xlsx)


data(XenaData)

XenaData%>%
  View()

unique(XenaData$XenaHostNames)

unique(XenaData$XenaCohorts)


#for the pancreatic cancer phenotype dataset
get_pancreatic_data_phenotypic <-
  XenaGenerate(subset = XenaHostNames=="gdcHub") %>% 
  XenaFilter(filterDatasets = "phenotype") %>% 
  XenaFilter(filterDatasets = "PAAD")

XenaQuery(get_pancreatic_data_phenotypic) %>%
  XenaDownload() -> xe_download_pancreatic_data_phenotypic


pancreatic_dataset_phenotypic = XenaPrepare(xe_download_pancreatic_data_phenotypic)
class(pancreatic_dataset_phenotypic)
paste0("The number of observations in the phenotypic dataset: ", nrow(pancreatic_dataset_phenotypic))



#counting the total number of observations and unique patients
pancreatic_dataset_phenotypic%>%
  dplyr::summarise(Number_of_observations=nrow(pancreatic_dataset_phenotypic),
                   Number_of_unique_patients=n_distinct(patient_id))


#counting the total number of males and females
pancreatic_dataset_phenotypic%>%
  group_by(gender.demographic)%>%
  dplyr::summarise(Number_of_observations=n(),
                    Number_of_unique_patients=n_distinct(patient_id))%>%
  adorn_totals(where = c("row")) %>%
  adorn_percentages("col") %>%
  adorn_pct_formatting(digits = 2) %>%
  adorn_ns("front")


#removing few normal tissue data
pancreatic_dataset_phenotypic <-
  pancreatic_dataset_phenotypic%>%
  filter(sample_type.samples!="Solid Tissue Normal", sample_type.samples!="")

pancreatic_dataset_phenotypic <-
  pancreatic_dataset_phenotypic%>%
  filter(sample_type.samples!="Metastatic", sample_type.samples!="")

pancreatic_dataset_phenotypic[!(is.na(pancreatic_dataset_phenotypic$sample_type.samples) | pancreatic_dataset_phenotypic$sample_type.samples==""), ]



#removing all the duplicates from the dataset
unduplicated_pancreatic_dataset_phenotypic=pancreatic_dataset_phenotypic[!duplicated(pancreatic_dataset_phenotypic$submitter_id.samples),]


#for the pancreatic cancer gene expression dataset
get_pancreatic_gene_expression <-
  XenaGenerate(subset = XenaHostNames=="gdcHub") %>% 
  XenaFilter(filterDatasets = "TCGA-PAAD.htseq_fpkm.tsv") %>%
  XenaFilter(filterDatasets = "PAAD")

XenaQuery(get_pancreatic_gene_expression) %>%
  XenaDownload() -> xe_download_pancreatic_data_gene_expression


raw_pancreatic_dataset_gene_expression = XenaPrepare(xe_download_pancreatic_data_gene_expression)
class(raw_pancreatic_dataset_gene_expression)
paste0("The number of observations in the gene_expression dataset: ", nrow(raw_pancreatic_dataset_gene_expression))


raw_pancreatic_dataset_gene_expression <- 
  raw_pancreatic_dataset_gene_expression[rowSums(raw_pancreatic_dataset_gene_expression[2:length(raw_pancreatic_dataset_gene_expression)])>0,]

#reading the file for gene code annotations
gene_code_annotations<-read.csv("gencode.v22.annotation.gene.probeMap.csv")

filtered_gene_code_annotations <- gene_code_annotations%>%
  dplyr::select("id","gene")


uncleaned_pancreatic_dataset_gene_expression <- 
  inner_join(raw_pancreatic_dataset_gene_expression,filtered_gene_code_annotations,
             by=c("Ensembl_ID"="id"))%>%
  dplyr::select("Ensembl_ID","gene",everything())%>%
  subset(select=-c(gene)) 



transposed_uncleaned_pancreatic_dataset_gene_expression<-t(uncleaned_pancreatic_dataset_gene_expression)
colnames(transposed_uncleaned_pancreatic_dataset_gene_expression)<-transposed_uncleaned_pancreatic_dataset_gene_expression[1,]
transposed_uncleaned_pancreatic_dataset_gene_expression<-transposed_uncleaned_pancreatic_dataset_gene_expression[-1,]
pancreatic_dataset_gene_expression<-as.data.frame(transposed_uncleaned_pancreatic_dataset_gene_expression)
pancreatic_dataset_gene_expression <- tibble::rownames_to_column(pancreatic_dataset_gene_expression, "Patient_Submitter_ID")

unduplicated_pancreatic_dataset_gene_expression<-pancreatic_dataset_gene_expression[!duplicated(pancreatic_dataset_gene_expression$Patient_Submitter_ID),]

unduplicated_pancreatic_dataset_gene_expression[,2:ncol(unduplicated_pancreatic_dataset_gene_expression)] <- 
  lapply(2:ncol(unduplicated_pancreatic_dataset_gene_expression), 
         function(x) as.numeric(unduplicated_pancreatic_dataset_gene_expression[[x]]))

pancreatic_dataset_combined <-
  inner_join(unduplicated_pancreatic_dataset_phenotypic,unduplicated_pancreatic_dataset_gene_expression,by=c("submitter_id.samples"="Patient_Submitter_ID"))


pancreatic_dataset_combined_binary <- pancreatic_dataset_combined%>%
  dplyr::select(submitter_id.samples,matches("^ENSG"))


pancreatic_dataset_combined_binary[pancreatic_dataset_combined_binary == 0] <- NA

pancreatic_dataset_combined_binary <- data.frame(pancreatic_dataset_combined_binary)



#importing the data file for the survival data
get_pancreatic_survival <-
  XenaGenerate(subset = XenaHostNames=="gdcHub") %>% 
  XenaFilter(filterDatasets = "survival") %>% 
  XenaFilter(filterDatasets = "PAAD")

XenaQuery(get_pancreatic_survival) %>%
  XenaDownload() -> xe_download_pancreatic_data_survival


raw_pancreatic_dataset_survival = XenaPrepare(xe_download_pancreatic_data_survival)
class(raw_pancreatic_dataset_survival)
paste0("The number of observations in the gene_expression dataset: ", nrow(raw_pancreatic_dataset_survival))

unduplicated_pancreatic_dataset_survival=raw_pancreatic_dataset_survival[!duplicated(raw_pancreatic_dataset_survival$sample),]

pancreatic_dataset_all <- 
  inner_join(pancreatic_dataset_combined_binary,unduplicated_pancreatic_dataset_survival,by=c("submitter_id.samples"="sample"))


#creating a new column for cut-off analysis
pancreatic_dataset_all<-
  pancreatic_dataset_all%>%
  mutate(censor=if_else((OS==1 & OS.time<1096),1,0))


pancreatic_dataset_all%>%
  dplyr::select(submitter_id.samples,OS,censor)%>%
  View()

#calculating the HR
HR_calculator_dataset <- pancreatic_dataset_all%>%
  dplyr::select(OS.time,OS,censor,matches("^ENSG"))%>% 
  dplyr::select(where(~mean(is.na(.)) < 0.75))


numOfCores <- detectCores()
cl <- makeCluster(numOfCores[1]-2)
 
registerDoParallel(cl)

HR_calculator_dataset_binary <-
  foreach (i = colnames(HR_calculator_dataset[4:ncol(HR_calculator_dataset)]), 
           .init = HR_calculator_dataset, .combine=cbind) %dopar% {
    out <- HR_calculator_dataset[,i,drop=FALSE]
    library(survminer)
    res.cut <- surv_cutpoint(HR_calculator_dataset, time = "OS.time", event = "censor",
                             variables = c(i))
    value<-res.cut$cutpoint[,1]
    out[[i]][out[[i]]<value] <- 0
    out[[i]][out[[i]]>=value] <- 1
    return(out)
  }

stopCluster(cl)

HR_calculator_dataset_binary <-
  HR_calculator_dataset_binary[,-4:-(ncol(HR_calculator_dataset))]

HR_calculator_dataset_binary[,4:ncol(HR_calculator_dataset_binary)] <-
   lapply(4:ncol(HR_calculator_dataset_binary),
          function(x) as.factor(HR_calculator_dataset_binary[[x]]))

HR_calculator_dataset_binary <-
  HR_calculator_dataset_binary%>%
   dplyr::select(OS.time, OS, censor,
                 all_of(colnames(HR_calculator_dataset_binary[, sapply(HR_calculator_dataset_binary, nlevels) > 1])))


#Calculating the HR with 3 years time cut off
numOfCores <- detectCores()

numOfCores <- detectCores()
cl <- makeCluster(numOfCores[1]-2)

registerDoParallel(cl)


HR_cutoff <- data.frame()
new_HR_cutoff <- data.frame()
new_HR_cutoff <-
  foreach (i = colnames(HR_calculator_dataset_binary)) %dopar% {
    library(survival)
    count_NAs <- sum(is.na(HR_calculator_dataset_binary[[i]]))
    for_count<-HR_calculator_dataset_binary[!is.na(HR_calculator_dataset_binary[[i]]),]
    count_data <- nrow(for_count)
    result_coxph_cutoff<-summary(coxph(Surv(OS.time, censor)~HR_calculator_dataset_binary[[i]],
                                              data=HR_calculator_dataset_binary))
    result_HR_cutoff=c(i,result_coxph_cutoff$coefficients[2],result_coxph_cutoff$sctest[3],count_NAs,count_data)
    HR_cutoff<-rbind(HR_cutoff,result_HR_cutoff)
    for_count <- HR_calculator_dataset_binary
    return(as.data.frame(HR_cutoff))
  }


stopCluster(cl)


new_HR_cutoff <- Map(as.data.frame, new_HR_cutoff)
HR_cutoff_genes <- data.frame()
HR_cutoff_genes <- rbindlist(new_HR_cutoff)

colnames(HR_cutoff_genes) <- c("gene_ID","Hazard_Ratio", "p-value", "Number of missing data",
                                      "Number of non-missing data")

HR_cutoff_genes <- 
  HR_cutoff_genes%>%
  filter(gene_ID!="OS.time", gene_ID!="OS", gene_ID!="censor")


HR_cutoff_genes<- HR_cutoff_genes[!duplicated(HR_cutoff_genes), ]

#Export the results
fwrite(HR_cutoff_genes, file="HR_results_cutoff_pancreatic.csv")



