import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow import keras

def __main__():

  # This program requires 5 arguments. If fewer are supplied, we print this usage information and exit(1)
  if len(sys.argv) < 5:
    print("\nUsage: python classify.py [data_dir] [data_domain] [smote]")
    print("\n  data_dir: Path to directory with data. Directory (for now) must contain gexp_matrix.npy, metadata_20210408.tsv, sample_names.txt, and gene_list.txt")
    print("\n  data_domain: Set of data to use. Options are major, AML, or B-ALL.")
    print("\n  smote: Whether or not to use SMOTE to balance the dataset. Options are True or False (or 0/1). SMOTE is a data augmentation algorithm that generates artificial samples of all minority classes. This may be useful if some classes are much larger than others.")
    exit(1)

  # process data directory, make sure there's a "/" at the end
  data_dir = sys.argv[1]
  if not data_dir.endswith("/"):
    data_dir = data_dir + "/"

  # assign args to vars
  model_type = "PLS"
  data_domain = sys.argv[2]
  smote = sys.argv[3]

  data_domain_keys = ["major","AML","B-ALL"]

  # if any args are not valid we print an error message and exit(1)
  if data_domain not in data_domain_keys:
    print("Invalid data domain. Domain must be one of the following: major, AML, B-ALL")
    print(data_domain)
    exit(1)
  if smote not in ["True","False","0","1"]:
    print("Invalid smote arg. Options are True/1 or False/0")
    print(smote)
    exit(1)

  # process smote arg to be python boolean literals
  if smote == "True" or smote == "1":
    smote = True
  elif smote == "False" or smote == "0":
    smote = False

  # process num_features arg to be python None literal or int literal
  if num_features == "None":
    num_features = None
  else:
    num_features = int(num_features)

  # Load data
  matrix_filename = data_dir + "gexp_matrix.npy"
  genes_filename = data_dir + "gene_list.txt"
  names_filename = data_dir + "sample_names.txt"
  metadata_filename = data_dir + "metadata.tsv"

  # Establish major data arrays
  X = np.load(matrix_filename) # names x genes
  genes = []
  with open(genes_filename,'r') as f:
    genes = f.readlines()
  names = []
  with open(names_filename,'r') as f:
    names = f.readlines()
  names = [name.strip('\n') for name in names]
  # run_name sample_name file lineage type_1 type_2 type_3 subtype_note attr_diagnosis attr_subtype_biomarkers subject_name SJ_ID
  metadata = pd.read_csv(metadata_filename, sep="\t")

  # drop duplicates from metadata
  metadata.drop_duplicates('run_name', keep='first', inplace=True)

  # Align X and y in case samples are not in the same order
  y_text = metadata.values
  permutation = [names.index(name) for name in y_text[:,0]]
  perm_idx = np.empty_like(permutation)
  perm_idx[permutation] = np.arange(len(permutation))
  y_text = y_text[perm_idx,:]

  # Major immunophenotypes
  label_count = 3
  major_types = {}
  major_types["AML"] = 0
  major_types["B-ALL"] = 1
  major_types["T-ALL"] = 2
  major_type_list = ["AML","B-ALL","T-ALL"]

  # Remove mixed, normal types
  deletions = []
  for i,major in enumerate(y_text[:,3]):
    if major not in major_types:
        deletions.append(i)
  X = np.delete(X, deletions, 0)
  y_text = np.delete(y_text, deletions, 0)
  names = [names[i] for i in range(len(names)) if i not in deletions]

  # Code to remove samples listed in bad_samples from training and testing
  """
  # Remove by name
  bad_samples = ["C9_ont_cdna","005_ont_cdna","126_ont_cdna_all"]
  deletions = [names.index(s) for s in bad_samples]
  X = np.delete(X, deletions, 0)
  y_text = np.delete(y_text, deletions, 0)
  names = [names[i] for i in range(len(names)) if i not in deletions]
  """

  # Filter out features where < 99.5% of samples have a nonzero value for them. This may or may not be commented out. Works well for major type classification but may be worse for minor type
  """
  # Remove genes with >99% 0 expression
  X = X[:,np.count_nonzero(X, axis=0)/float(X.shape[0]) > 0.995]
  """

  # Scale data using sklearn StandardScaler
  scaler = StandardScaler()
  X = scaler.fit_transform(X)

  # finalize y
  y = np.array([major_types[s] for s in y_text[:,3]])

  # Set up AML classification
  aml_indices = [i for i in range(y_text.shape[0]) if y_text[i,3] == "AML" and y_text[i,4] != "Exclude"]
  X_aml = X[aml_indices]
  y_text_aml = y_text[aml_indices]
  aml_type_list = ["Core binding factor","KMT2Ar","Other"]
  aml_dict = dict(zip(aml_type_list,[i for i in range(len(aml_type_list))]))
  y_aml = np.array([aml_dict[subtype] for subtype in y_text_aml[:,4]])

  # Set up B-ALL classification
  ball_indices = [i for i in range(y_text.shape[0]) if y_text[i,3] == "B-ALL" and y_text[i,4] != "unknown"]
  X_ball = X[ball_indices]
  y_text_ball = y_text[ball_indices]
  ball_type_list = ["ETV6-RUNX1","Hyperdiploid / Near haploid","Low hypodiploid","KMT2A","Ph / Ph-like","TCF3-PBX1","Other"]

  # dictionary of main B-ALL subtypes (at least 6 samples)
  ball_dict = dict(zip(ball_type_list,[i for i in range(len(ball_type_list))]))

  # map all BALL subtypes to more condensed ball_type_list
  ball_simplify = {}
  for s in ball_type_list:
    ball_simplify[s] = s
  ball_simplify["KMT2Ar"] = "KMT2A"
  ball_simplify["KMT2A+E487"] = "KMT2A"

  # finalize y array for B-ALL
  y_ball = np.array([ball_dict[ball_simplify[subtype]] for subtype in y_text_ball[:,4]])

  # Non-illumina indices, which are the only ones we test on
  ni_indices = [i for i in range(y_text.shape[0]) if not y_text[i,0].startswith("SJ")]
  ni_aml_indices = [i for i in range(y_text_aml.shape[0]) if not y_text_aml[i,0].startswith("SJ")]
  ni_ball_indices = [i for i in range(y_text_ball.shape[0]) if not y_text_ball[i,0].startswith("SJ")]

  # based on input parameters, specify which versions (major, AML, BALL) of our data will be passed to the model
  final_X = X if data_domain == "major" else X_aml if data_domain == "AML" else X_ball
  final_y = y if data_domain == "major" else y_aml if data_domain == "AML" else y_ball
  final_names = names if data_domain == "major" else [names[i] for i in aml_indices] if data_domain == "AML" else [names[i] for i in ball_indices]
  final_indices = ni_indices if data_domain == "major" else ni_aml_indices if data_domain == "AML" else ni_ball_indices
  final_codes = major_type_list if data_domain == "major" else aml_type_list if data_domain == "AML" else ball_type_list

  # Final call to methods for classification
  predictions = get_predictions_plsda(final_X, final_y, final_names, final_indices, [i for i in range(5,11)],do_smote=smote,n_features=num_features)

  # print out final results of training and testing
  print_accuracies(predictions,final_y,final_names,final_codes)


# names is only the names at the same indices as X and y
# Important - make sure names is properly filtered before passing
def generate_datasplits(X,y,names,leave_out_num,exclude_samples=None):

    # this isn't great form -- ideally exclude samples would be a global var
    # but this is serviceable for now
    # samples in exclude_samples are excluded from training sets
    if exclude_samples == None:
      exclude_samples = ["005_ont_cdna","009_ont_cdna","126_ont_cdna_all","252_ont_cdna","C9_ont_cdna","E3_ont_cdna"]

    # pull out one sample used for testing, reshape to fit classification model
    X_test = X[leave_out_num].reshape(1,-1)

    # assemble array of sample indices that will be excluded from training
    # all members of exclude_samples and any sample that shares an ID with the test sample is excluded
    deletions = [leave_out_num]
    # exclude anything in exclude_samples
    if exclude_samples is not None: # in case the above default value of exclude_samples is ever commented out
      for sample in exclude_samples:
        if sample in names:
          deletions.append(names.index(sample))
    # exclude anything that shares an ID with the test sample
    test_name = names[leave_out_num]
    test_id = test_name.split("_")[0]
    seen = False
    for i,name in enumerate(names):
      if name.split("_")[0] == test_id:
        deletions.append(i)
        seen = True
      else:
        # there should only ever be 1 variant of a given sample, and sample names are in order
        # this slightly improves performance because once we see all samples with the same ID we never see another
        if seen:
          break

    # convert deletions to a set in case a sample shares an ID with the test sample and is in exclude_samples
    deletions = list(set(deletions))

    # remove excluded samples from training set
    X_train = np.delete(X, deletions, 0)
    y_train = np.delete(y, deletions, 0)

    return X_train, y_train, X_test


# main method for a PLS model
def get_predictions_plsda(X,y,names,ni_indices,nums_components,do_smote=False,n_features=None):

    # number of classes is used when generating one-vs-one and one-vs-all models
    num_y = np.max(y) + 1

    # array of predictions that will be returned
    predictions = []

    # if n_features is specified, reduce X features to only the n_features most important using Sklearn's ExtraTreesClassifier
    if n_features:
        forest = ExtraTreesClassifier(n_estimators=250, max_depth=5, random_state=1)
        forest.fit(X, y)
        importances = forest.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in forest.estimators_],
            axis=0
        )
        indices = np.argsort(importances)[::-1]
        indices = indices[:n_features]
        X = X[:,indices]

    # iterate over and test on every non-illumina sample
    for idx in ni_indices:

        # divide X into leave-one-out training and testing sets
        X_train, y_train, X_test = generate_datasplits(X,y,names,idx)

        # if smote is True, augment the training set
        if do_smote:
          smote = SMOTE(n_jobs=-1)
          X_train, y_train = smote.fit_resample(X_train, y_train)

        # array of PLSRegression models that will be fed into final SVC discriminant analysis model
        models = []

        # generate a one-vs-all model for each class
        for cls in range(num_y):

            # one-vs-all y array labels the highlighted class with a 0 and anything else with a 1
            y_temp = np.array([0 if c == cls else 1 for c in y_train])

            # only proceed if the y array is not homogenous
            if sum(y_temp) > 0:

                # for each value of num_components
                for num_components in nums_components:

                    # fit model on training set and append to models array
                    model = PLSRegression(n_components=num_components)
                    model.fit(X_train, y_temp)
                    models.append(model)

        # generate one-vs-one model for each combo of classes
        for cls1 in range(num_y - 1):
            for cls2 in range(cls1+1, num_y):

                # single out sample indices that are either of our two highlighted classes
                temp_idxs = np.array([i for i in range(X_train.shape[0]) if (y_train[i] == cls1 or y_train[i] == cls2)])

                # one vs one X and y only exist at our temp indices
                X_temp = X_train[temp_idxs]
                # label class A with a 0 and class B with a 1
                y_temp = np.array([0 if c == cls1 else 1 for c in y_train[temp_idxs]])

                # only proceed if the y array is not homogenous
                if sum(y_temp) > 0:

                    # for each value of num_components
                    for num_components in nums_components:

                        # fit model on training set and append to models array
                        model = PLSRegression(n_components=num_components)
                        model.fit(X_temp, y_temp)
                        models.append(model)

        # training set for our SVC discriminant analysis model
        # query each of our models on the training set again
        train_out = np.zeros((X_train.shape[0],len(models)))
        for i,model in enumerate(models):
            train_out[:,i] = model.predict(X_train).flatten()

        # testing set for our SVC discriminant analysis model
        # query each of our models on the testing sample
        test_out = np.zeros(len(models))
        for i,model in enumerate(models):
            test_out[i] = model.predict(X_test).flatten()

        # create our SVC discriminant analysis model
        clf = SVC(probability=True)
        # fit on our newly created training set
        clf.fit(train_out,y_train)

        # final guess and probability obtained by querying the SVC on our newly created test set
        guess = clf.predict(test_out.reshape(1,-1))[0]
        probs = clf.predict_proba(test_out.reshape(1,-1))[0]
        predictions.append((guess, probs[guess]))

    return predictions

# prints accuracies of model guesses
# prints model output for every sample
def print_accuracies(guesses, truths, names, codes):

    n_right = 0
    n_right_high_prob = 0
    n_high_prob = 0
    for guess,truth in zip(guesses, truths):
        val = guess[0]
        prob = guess[1]
        if val == truth:
            n_right += 1
            if prob > 0.8:
                n_right_high_prob += 1
        if prob > 0.8:
            n_high_prob += 1
    print(str(n_right) + " / " + str(len(guesses)) + " correct")
    print(str(n_right/float(len(guesses))) + " correct")
    print(str(n_right_high_prob) + " / " + str(n_high_prob) + " correct with high probability")
    print(str(n_right_high_prob/float(n_high_prob)) + " correct with high probability")
    wrongs = [i for i in range(len(guesses)) if guesses[i][0] != truths[i]]
    print("\nWrong:")
    for i in wrongs:
        print(names[i] + ": " + codes[truths[i]] + " -> " + codes[guesses[i][0]] + " w/ p = " + str(guesses[i][1]))
    print("\nAll:")
    for i,(v,p) in enumerate(guesses):
        print(names[i] + "\t" + codes[v] + "\t" + str(p))


if __name__ == "__main__":
  __main__()
