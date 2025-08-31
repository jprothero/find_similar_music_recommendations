from flask import Flask, jsonify, request
import pickle as p

from ease_recommender import *

if __name__ == '__main__':
    app = Flask(__name__)

    print("loading cache data...")
    D = p.load(open("cached_data/spotify_preprocessed.p", "rb"))

    # TODO: add track/album level recommendations, starting with artist level

    # track2idx = D["track2idx"]
    # album2idx = D["album2idx"]
    artist2idx = D["artist2idx"]

    playlist_indices = D["playlist_indices"]
    # track_indices = D["track_indices"]
    # album_indices = D["album_indices"]
    artist_indices = D["artist_indices"]
    
    print("building csr matrices...")
    row = playlist_indices
    col = artist_indices
    data = np.ones_like(playlist_indices, dtype=np.int64)
    
    D["artist_mat"] = csr_matrix((data, (row, col)))
    print("done")

    def check_if_all_terms_in_str(q, terms):
        for term in terms:
            if term not in q:
                return False

        return True

    def get_cat2idx(category_type):
        if category_type == "track":
            return track2idx
        elif category_type == "album":
            return album2idx
        elif category_type == "artist":
            return artist2idx
        else:
            raise NotImplementedError

    def find_matches_using_terms(terms, cat2idx, max_matches=1):
        matches = []
        for name in cat2idx.keys():
            if check_if_all_terms_in_str(name, terms):
                matches.append(name)
                if len(matches) > max_matches:
                    raise Exception(f"Exceeded max_matches ({max_matches})")

        if len(matches) == 0:
            raise Exception("No matches found")

        return matches

#     @app.route('/find_by_terms', methods=['GET'])
#     def find_by_terms():
#         # Get the JSON data sent with the POST request
#         data = request.get_json()

#         terms_string = request.args.get('terms', '')
#         terms = [term.strip() for term in terms_string.split(",")]
#         terms = filter(terms, key=lambda x: len(x) > 0)

#         category_type = data["category_type"]
#         assert category_type in ["track", "album", "artist"]

#         matches = find_matches_using_terms(terms, category_type)

#         return jsonify({
#             'matches': matches,
#         })

    @app.route('/find_similar', methods=['GET'])
    def find_similar():
        # Get the JSON data sent with the POST request
        data = request.get_json()

        terms_string = request.args.get('terms', '')
        terms = [term.strip() for term in terms_string.split(",")]
        terms = filter(terms, key=lambda x: len(x) > 0)

    #     category_type = data["category_type"]
    #     assert category_type in ["track", "album", "artist"]

        # TODO: add track and album level

        if category_type == "track":
            raise NotImplementedError
        elif category_type == "album":
            raise NotImplementedError
        elif category_type == "artist":
            mat = D["artist_mat"]
        else:
            raise NotImplementedError

        category_type = "artist"
        lambda_ = 100
        top_k = 10

        cat2idx = get_cat2idx(category_type)

        match_name = find_matches_using_terms(terms, cat2idx)[0]
        match_idx = cat2idx[match_name]

        similarity_scores = calculate_ease_for_item_logistic(mat, match_idx, lambda_)

        top_k_matches = np.argsort(-similarity_scores)[:top_k]

        return ", ".join(top_k_matches)
        
#         return jsonify({
#             'top_k_matches': top_k_matches
#         })

    app.run(port=1234)