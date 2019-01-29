def get_dataset_path_list(dataset_path, sub_str=None):
  dataset_path_list = []
  for root_path, dir_names, file_names in os.walk(dataset_path):
    for file_name in file_names:
      file_path = os.path.join(root_path, file_name)
      if sub_str:
        if file_path.find(sub_str) == -1:
          continue
      dataset_path_list.append(file_path)

  return dataset_path_list
