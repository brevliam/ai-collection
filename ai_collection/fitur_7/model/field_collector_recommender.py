import pandas as pd
import math

class FieldCollectorRecommender():
  __collector_df = None

  def __init__(self):
    return None
  
  def set_collector_dataset(self, dataset_path):
    self.collector_df = pd.read_csv(dataset_path)

  def recommend(self, debtor_data):
    debtor_data = {key: value for key, value in debtor_data.items()}

    nearest_collector = []
    for index, row in self.collector_df.iterrows():
      collector_coord = row['collector_location_coord']
      debtor_coord = debtor_data.get('debtor_location_coord')
      distance = self.compute_distance(debtor_coord, collector_coord)

      data = (row['collector_nik'], distance, row['workload_score'], row['collector_vehicle'])

      if len(nearest_collector) == 0:
        nearest_collector.append(data)
      else:
        nearest_distance = nearest_collector[0][1]
        if nearest_distance == distance:
          nearest_collector.append(data)
        elif nearest_distance > distance:
          nearest_collector = [data]

    min_workload_collector = []
    for data in nearest_collector:
      if len(min_workload_collector) == 0:
        min_workload_collector.append(data)
      else:
        recent_min_workload = min_workload_collector[0][2]
        if recent_min_workload == data[2]:
          min_workload_collector.append(data)
        elif recent_min_workload > data[2]:
          min_workload_collector = [data]

    recommendation = None
    for data in min_workload_collector:
      if data[3] == 'Punya':
        recommendation = data[0]
        break

    if recommendation == None:
      recommendation = min_workload_collector[0][0]

    field_collector = self.collector_df[self.collector_df['collector_nik'] == recommendation]

    return field_collector.to_dict(orient='records')[0]


  def compute_distance(self, debtor_coord, collector_coord):
    """
    Calculate the great circle distance in kilometers between two points
    on the Earth's surface specified in decimal degrees of latitude and longitude.
    """
    debtor_coord = debtor_coord.split(':')
    collector_coord = collector_coord.split(':')

    lat1 = float(debtor_coord[0])
    lon1 = float(debtor_coord[1])
    lat2 = float(collector_coord[0])
    lon2 = float(collector_coord[1])

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Radius of the Earth (mean value) in kilometers
    radius = 6371.0

    # Calculate the distance
    distance = radius * c

    return distance
