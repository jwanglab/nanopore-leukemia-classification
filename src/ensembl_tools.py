import csv

# ENSEMBL data mart dump with the following fields in CSV format:
#
# Gene stable ID
# Gene stable ID version
# Transcript stable ID
# Transcript stable ID version
# Gene name
# Gene Synonym
# Chromosome/scaffold name
# Gene start (bp)
# Gene end (bp)
# Gene description
# Transcript start (bp)
# Transcript end (bp)
# Transcript name
# HGNC symbol

class ensembl:

  def __init__(self, table_file):
    self.tx_id2gene_id = {}
    self.synonym2gene_id = {}
    self.gene_id2gene_name = {}
    self.gene_name2gene_id = {}
    self.gene_location = {}
    self.tx_version2tx_id = {}
    reader = csv.reader(open(table_file), delimiter='\t')
    header = next(reader, None)
    tx_idx = header.index("Transcript stable ID") if "Transcript stable ID" in header else header.index("Ensembl Transcript ID")
    txv_idx = header.index("Transcript stable ID version") if "Transcript stable ID version" in header else None
    gn_idx = header.index("Gene stable ID") if "Gene stable ID" in header else header.index("Ensembl Gene ID")
    gnv_idx = header.index("Gene stable ID version") if "Gene stable ID version" in header else None
    name_idx = header.index("Gene name") if "Gene name" in header else header.index("Associated Gene Name")
    syn_idx = header.index("Gene Synonym") if "Gene Synonym" in header else None
    chrom_idx = header.index("Chromosome/scaffold name") if "Chromosome/scaffold name" in header else header.index("Chromosome Name")
    start_idx = header.index("Gene start (bp)") if "Gene start (bp)" in header else header.index("Gene Start (bp)")
    end_idx = header.index("Gene end (bp)") if "Gene end (bp)" in header else header.index("Gene End (bp)")
    for row in reader:
      self.tx_id2gene_id[row[tx_idx]] = row[gn_idx]
      if syn_idx is not None:
        self.synonym2gene_id[row[syn_idx]] = row[gn_idx]
      self.gene_id2gene_name[row[gn_idx]] = row[name_idx]
      self.gene_name2gene_id[row[name_idx]] = row[gn_idx]
      self.gene_location[row[gn_idx]] = (row[chrom_idx], int(row[start_idx]), int(row[end_idx])) # chrom, st, en
      if txv_idx is not None:
        self.tx_version2tx_id[row[txv_idx]] = row[tx_idx]

  def get_gene_id(self, obj):
    if obj in self.tx_version2tx_id:
      return self.tx_id2gene_id[self.tx_version2tx_id[obj]]
    if obj in self.tx_id2gene_id:
      return self.tx_id2gene_id[obj]
    if obj in self.synonym2gene_id:
      return self.synonym2gene_id[obj]
    if obj in self.gene_id2gene_name:
      return obj # it IS a gene ID
    if obj in self.gene_name2gene_id:
      return self.gene_name2gene_id[obj]
    return None

  def get_gene_name(self, obj):
    gene_id = self.get_gene_id(obj)
    if gene_id is None:
      return None
    return self.gene_id2gene_name[gene_id]
