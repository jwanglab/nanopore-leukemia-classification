import sys
import os.path
import argparse
import numpy as np
import csv
from collections import defaultdict, Counter
from ensembl_tools import ensembl

def main(fof, ensembl_files, matrix_file, gene_list_file, sample_names_file):
  ensembl_tables = [ensembl(f) for f in ensembl_files]

  data = {}
  target_metadata = {}
  shared_rids = None
  sj_ids = {}

  with open(fof) as f:
    for run_name, sample_name, exp_file, lineage, type_1, type_2, type_3, subtype_note, attr_diagnosis, attr_subtype_biomarkers, subject_name, SJ_ID in csv.reader(f, delimiter='\t'):
      if sample_name == "sample_name": # header
        continue

      data[run_name] = defaultdict(int)
      print(run_name, exp_file)

      if exp_file[exp_file.rindex('.')+1:] in ["txt", "cts", "sf"]: # probably minnow, Salmon, or TARGET data (with header)
        if not os.path.exists(exp_file):
          print(f"ERROR: {exp_file} does not exist! Skipping it")
          del data[run_name]
          continue
        with open(exp_file) as ef:
          reader = csv.reader(ef, delimiter='\t')
          header = next(reader, None)
          if header is None: # empty file
            print(f"ERROR: no expression data in {exp_file}, skipping it")
            del data[run_name]
            continue
          #gene.expression (TARGET): gene	raw_counts	median_length_normalized	RPKM
          #minnow: transcript	reads	TPM
          #salmon: Name	Length	EffectiveLength	TPM	NumReads
          #gene.quantification.txt (TARGET/AML): gene	raw_count	rpkm
          for field_name in ["gene", "transcript", "Name"]:
            if field_name in header:
              gene_idx = header.index(field_name)
              break
          else:
            gene_idx = None
          for field_name in ["RPKM", "TPM", "rpkm", "fpkm_normalized"]:
            if field_name in header:
              ct_idx = header.index(field_name)
              break
          else:
            print(header)
            ct_idx = None
          n_rows = 0
          skipped = 0
          for row in reader:
            n_rows += 1
            # check ensembl tables in order for a valid conversion
            for et in ensembl_tables:
              if '|' in row[gene_idx]:
                for part in row[gene_idx].split('|')[::-1]: # usually something like "IL15|ENSG00000164136"
                  eid = et.get_gene_id(part)
                  if eid is not None:
                    break
              else:
                eid = et.get_gene_id(row[gene_idx])
              if eid is not None:
                gene_name = et.get_gene_name(eid)
                break
            if eid is None:
              #print(f"WARNING: cannot convert {row[gene_idx]} to a valid ENSEMBL gene ID")
              skipped += 1
              #sys.exit(1)
              continue
            data[run_name][gene_name] += float(row[ct_idx]) # multiple IDs or transcripts might contribute to the same gene, so add them
          if skipped > 0:
            print(f"WARNING: skipped {skipped} of {n_rows} ({skipped/n_rows*100:.2f}%) expression rows in {exp_file}")
      else:
        print("ERROR: unknown extension '{}' of file '{}'".format(td[td.rindex('.'):], td))
        return

  run_names = sorted(list(set([n for n in data])))
  eids = sorted(list(set([eid for name in data for eid in data[name]])))
  print(f"{len(run_names)} samples processed")
  print(f"total genes: {len(eids)}")

  # output ordered list of samples
  with open(sample_names_file, 'w') as f:
    f.write('\n'.join(run_names) + '\n')
  # output ordered list of ENSEMBL gene IDs
  with open(gene_list_file, 'w') as f:
    f.write('\n'.join(eids) + '\n')

  # save grid data if necessary
  matrix = np.array([[data[n][gn] for gn in eids] for n in run_names])
  np.save(matrix_file, matrix)

if __name__ == "__main__":
  parser = argparse.ArgumentParser("Build joint TARGET/nanopore gene expression matrix")
  parser.add_argument("fof", help="File of sample names and expression files")
  parser.add_argument("matrix", help="Output file for expression matrix (.npy)")
  parser.add_argument("gene_list", help="Output file gene ID list (.txt)")
  parser.add_argument("sample_names", help="Output file sample name list (.txt)")
  parser.add_argument("ensembl", help="Ensembl table(s), will be checked in order", nargs='+')
  args = parser.parse_args()
  main(args.fof, args.ensembl, args.matrix, args.gene_list, args.sample_names)
