import pandas as pd
import random
import math


# takes 2 row series and calculates the distances between them
def euclidean_dist(a: pd.Series, b: pd.Series):
    diff = a.sub(other=b)
    squares = diff ** 2
    dist = 0

    for feature_distance in squares:
        if not math.isnan(feature_distance):
            dist += feature_distance

    return math.sqrt(dist)


# takes copy of dataframe; returns initialized centroid array
def choose_centroids(data_copy: pd.DataFrame):
    new_centroids = []

    # randomly picks k centroids
    for i in range(0, k):
        distance_scores = []

        # picks furthest centroid from each other if the first one has been picked; else picks a random initial point
        if i != 0:
            for j in new_centroids:
                distances = []

                # for j existing centroids, compare to all other points and selects from all of j for next centroid
                for row in data_copy.iterrows():
                    distances.append((euclidean_dist(j, row[1]), row[0]))

                distances.sort()
                distance_scores.append(distances[-1])

            distance_scores.sort()
            centroid_index = distance_scores[-1][1]

        else:
            centroid_index = random.randrange(num_rows)

        # drops centroid from copied dataframe to avoid duplicates
        data_copy.drop(labels=centroid_index, axis=0, inplace=True)

        # appends the newly selected centroid to the list
        new_centroids.append(data.iloc[centroid_index])

    return new_centroids


def assign_centroids():
    cluster_ids = []        # array for storing column output
    cluster_dict = {}       # dict for mapping centroid IDs (i.e. 89, 102, 34, etc.) to (0, 1, 2, ..., k)
    counter = 0

    for i in centroids:
        if i.name is None:
            i.name = counter
        cluster_dict[i.name] = counter
        counter += 1        # crude way of assigning centroid IDs

    for row in data.iterrows():
        distances = []

        for j in centroids:
            dist = euclidean_dist(row[1], j)
            if dist != 0:
                distances.append((dist, j.name))

        distances.sort()
        cluster_ids.append(cluster_dict[distances[0][1]])

    # inserts cluster assignment column;
    # if column already exists, catches exception and removes the column before insertion
    try:
        data.insert(6, "ClusterID", cluster_ids)
    except ValueError:
        data.drop(columns="ClusterID", axis=1, inplace=True)
        data.insert(6, "ClusterID", cluster_ids)
    except IndexError:
        data.drop(columns="ClusterID", axis=1, inplace=True)
        data.insert(6, "ClusterID", cluster_ids)
    return cluster_ids


def recalculate_clusters():
    # for k centroids, take the mean of all values belonging to the centroid and make that point the new centroid
    for i in range(0, k):
        cluster = pd.DataFrame()
        for item in data.iterrows():
            if item[1].loc['ClusterID'] == i:
                cluster = cluster.append(other=item[1])
        centroids[i] = cluster.mean()


data = pd.read_csv("data/fire_data_2011.csv")

# uses a dict to convert from tree genus i.e. "Pinu", "Pice",... to 0, 1,...
counter = 0
tree_count_dict = {}
for i in data.iterrows():
    try:
        tree_count_dict[i[1]["tree_genus"]]
    except KeyError:
        tree_count_dict[i[1]["tree_genus"]] = counter
        counter += 1

data = data.copy().replace(to_replace=tree_count_dict)
print(data)

k = 7
num_rows = data.iloc[-1].name       # gets label of the last row to figure out how many instances are in the data

# giving temporary copy of data so selected values can be removed so there aren't duplicate centroids
centroids = choose_centroids(data.copy())

cluster_assignments = []
unchanged_iteration_count = 0

for iterations in range(0, 100):
    print("Clustering Progress: [", iterations + 1, "/ 100 ]")

    # update previous cluster assignments; reassign cluster IDs and recalculate centroids
    previous_assignments = cluster_assignments.copy()
    cluster_assignments = assign_centroids()
    recalculate_clusters()

    # checks if cluster assignments have changed from one iteration to another
    if previous_assignments == cluster_assignments and len(previous_assignments) > 0:
        unchanged_iteration_count += 1
    else:
        unchanged_iteration_count = 0

    # if cluster assignments haven't changed in 3 iterations, break from loop and exit
    if unchanged_iteration_count > 3:
        print("Exiting early: cluster assignments haven't changed in 3 iterations")
        break

print("\nCluster Counts ( k =", k, "):")
for i in range(0, k):
    print("Cluster", i + 1, ": ", cluster_assignments.count(i))

print("\n\n", data)

data.to_csv("./data/fire_data_2011_clustered.csv")
