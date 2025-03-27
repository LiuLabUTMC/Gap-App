
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
library(readxl)
library(writexl)
library(xlsx)



data(XenaData)

XenaData%>%
  View()

unique(XenaData$XenaHostNames)
unique(XenaData$XenaCohorts)


#getting the datasets

#for the GTEX phenotype dataset
get_gtex_data_phenotypic <-
  XenaGenerate(subset = XenaHostNames=="toilHub") %>%
  XenaFilter(filterDatasets = "GTEX") %>%
  XenaFilter(filterDatasets = "TcgaTargetGTEX_phenotype")

XenaQuery(get_gtex_data_phenotypic) %>%
  XenaDownload() -> xe_download_gtex_data_phenotypic

gtex_dataset_phenotypic = XenaPrepare(xe_download_gtex_data_phenotypic)
class(gtex_dataset_phenotypic)
paste0("The number of observations in the phenotypic dataset: ", nrow(gtex_dataset_phenotypic))

gtex_pancreatic_dataset_phenotypic <- 
  gtex_dataset_phenotypic%>%
  filter(`_primary_site`=="Pancreas")

paste0("The number of observations in the phenotypic dataset: ", nrow(gtex_pancreatic_dataset_phenotypic))

#Adding a new column: tumor vs non-tumor
gtex_pancreatic_dataset_phenotypic <-
  gtex_pancreatic_dataset_phenotypic%>%
  mutate(tumor_NoTumor = ifelse(`_sample_type`=="Primary Tumor", 1, 0))

gtex_pancreatic_dataset_phenotypic_final <-
  gtex_pancreatic_dataset_phenotypic%>%
  dplyr::select(sample,`_primary_site`,`_sample_type`,`_gender`,`_study`,tumor_NoTumor)

colnames(gtex_pancreatic_dataset_phenotypic_final) <- c("Patient_ID","Primary_site","Sample_type","Gender","Study","Tumor_NoTumor")

unduplicated_pancreatic_dataset_phenotypic=gtex_pancreatic_dataset_phenotypic_final[!duplicated(gtex_pancreatic_dataset_phenotypic_final$Patient_ID),]

patients <- c(unduplicated_pancreatic_dataset_phenotypic$Patient_ID)

#for the GTEX gene expression dataset
get_gtex_gene_expression <-
  XenaGenerate(subset = XenaHostNames=="toilHub") %>%
  XenaFilter(filterDatasets = "GTEX") %>%
  XenaFilter(filterDatasets = "TcgaTargetGtex_rsem_gene_fpkm")

XenaQuery(get_gtex_gene_expression) %>%
  XenaDownload() -> xe_download_gtex_gene_expression

Sys.setenv("VROOM_CONNECTION_SIZE" = 262144 * 2)

gtex_dataset_gene_expression = XenaPrepare(xe_download_gtex_gene_expression)
class(gtex_dataset_gene_expression)
paste0("The number of observations in the gene expression dataset: ", nrow(gtex_dataset_gene_expression))

patients_gtex <- c("sample",unduplicated_pancreatic_dataset_phenotypic$Patient_ID)
gtex_dataset_gene_expression <- gtex_dataset_gene_expression[,colnames(gtex_dataset_gene_expression) %in% patients_gtex] 


gtex_dataset_gene_expression$sample <- gsub("\\..*","",as.character(gtex_dataset_gene_expression$sample))

#reading the file for gene code annotations
gene_code_annotations<-read.csv("gencode.v22.annotation.gene.probeMap.csv")

filtered_gene_code_annotations <- gene_code_annotations%>%
  dplyr::select("id","id.1","gene")

uncleaned_dataset_gene_expression <- 
  inner_join(gtex_dataset_gene_expression,filtered_gene_code_annotations,
             by=c("sample"="id.1"))%>%
  dplyr::select("sample","gene",everything())%>%
  subset(select=-c(gene,id))  



transposed_uncleaned_dataset_gene_expression<-t(uncleaned_dataset_gene_expression)


colnames(transposed_uncleaned_dataset_gene_expression)<-transposed_uncleaned_dataset_gene_expression[1,]
transposed_uncleaned_dataset_gene_expression<-transposed_uncleaned_dataset_gene_expression[-1,]
new_dataset_gene_expression<-as.data.frame(transposed_uncleaned_dataset_gene_expression)
new_dataset_gene_expression <- tibble::rownames_to_column(new_dataset_gene_expression, "Patient_Submitter_ID")

#removing all the duplicates from the dataset
unduplicated_pancreatic_dataset_gene_expression<-new_dataset_gene_expression[!duplicated(new_dataset_gene_expression$Patient_Submitter_ID),]

pancreatic_dataset_combined <-
  inner_join(unduplicated_pancreatic_dataset_phenotypic,unduplicated_pancreatic_dataset_gene_expression,by=c("Patient_ID"="Patient_Submitter_ID"))

pancreatic_dataset_all <- pancreatic_dataset_combined


#for the pancreatic cancer gene expression dataset
get_gtex_gene_expression_count <-
  XenaGenerate(subset = XenaHostNames=="toilHub") %>% 
  XenaFilter(filterDatasets = "GTEX") %>% 
  XenaFilter(filterDatasets = "TcgaTargetGtex_gene_expected_count")

XenaQuery(get_gtex_gene_expression_count) %>%
  XenaDownload() -> xe_download_gtex_data_gene_expression_count

Sys.setenv("VROOM_CONNECTION_SIZE" = 262144 * 2)

gtex_gene_expression_count = XenaPrepare(xe_download_gtex_data_gene_expression_count)
class(gtex_gene_expression_count)
paste0("The number of observations in the gene_expression dataset: ", nrow(gtex_gene_expression_count))


gtex_gene_expression_count <- gtex_gene_expression_count[,colnames(gtex_gene_expression_count) %in% patients_gtex] 

new_gtex_gene_expression_count <- gtex_gene_expression_count[,2:ncol(gtex_gene_expression_count)]
new_gtex_gene_expression_count <- (2^new_gtex_gene_expression_count)-1

clean_gtex_gene_expression_count<-cbind(gtex_gene_expression_count%>%
                                    dplyr::select("sample"),
                                    new_gtex_gene_expression_count)

clean_gtex_gene_expression_count$sample <- gsub("\\..*","",as.character(clean_gtex_gene_expression_count$sample))


uncleaned_dataset_gene_expression_count <- 
  inner_join(clean_gtex_gene_expression_count,filtered_gene_code_annotations,
             by=c("sample"="id.1"))%>%
  dplyr::select("sample","gene",everything())%>%
  subset(select=-c(gene,id))  


transposed_uncleaned_pancreatic_dataset_gene_expression_count <- 
  t(uncleaned_dataset_gene_expression_count)

colnames(transposed_uncleaned_pancreatic_dataset_gene_expression_count) <- 
  transposed_uncleaned_pancreatic_dataset_gene_expression_count[1,]

transposed_uncleaned_pancreatic_dataset_gene_expression_count <- 
  transposed_uncleaned_pancreatic_dataset_gene_expression_count[-1,]

pancreatic_dataset_gene_expression_count <- 
  as.data.frame(transposed_uncleaned_pancreatic_dataset_gene_expression_count)

pancreatic_dataset_gene_expression_count <- tibble::rownames_to_column(pancreatic_dataset_gene_expression_count,
                                                                       "Patient_Submitter_ID")


unduplicated_pancreatic_dataset_gene_expression_count <- 
  pancreatic_dataset_gene_expression_count[!duplicated(pancreatic_dataset_gene_expression_count$Patient_Submitter_ID),]

unduplicated_pancreatic_dataset_gene_expression_count[,2:ncol(unduplicated_pancreatic_dataset_gene_expression_count)] <- 
  lapply(2:ncol(unduplicated_pancreatic_dataset_gene_expression_count), 
         function(x) as.numeric(unduplicated_pancreatic_dataset_gene_expression_count[[x]]))


#Merging
pancreatic_dataset_gene_expression_count <-
  inner_join(pancreatic_dataset_all%>%
               dplyr::select("Patient_ID"),
             unduplicated_pancreatic_dataset_gene_expression_count,
             by=c("Patient_ID"="Patient_Submitter_ID"))


data_normal_count <- pancreatic_dataset_gene_expression_count%>%
  dplyr::select(-c("Patient_ID"))

data_tumorVSnotumor_binary <- pancreatic_dataset_all%>%
  dplyr::select(c("Tumor_NoTumor"))


new_data_tumorVSnotumor_binary<-t(data_tumorVSnotumor_binary)
new_data_normal_count<-t(data_normal_count)

y <- DGEList(counts=new_data_normal_count, genes=rownames(new_data_normal_count), 
             group = new_data_tumorVSnotumor_binary)
y <- estimateDisp(y)
et <- exactTest(y)

result_edgeR <- as.data.frame(topTags(et, n=nrow(new_data_normal_count)))

significant_result_edgeR <- result_edgeR%>%
   filter(FDR<0.05 & abs(logFC)>1)


write.xlsx(significant_result_edgeR, file = "significant_edgeR_results_TCGA_GTEX.xlsx",
           sheetName="DEGs", row.names=FALSE)

