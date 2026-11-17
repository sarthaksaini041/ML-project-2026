"""Small connectivity/schema smoke test. Makes only a few public API requests."""
import argparse
from src.cf_api import CodeforcesAPI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handle", default="tourist")
    args = parser.parse_args()

    api = CodeforcesAPI(min_interval_seconds=2.1)
    profile = api.get("user.info", {"handles": args.handle})[0]
    rating = api.get("user.rating", {"handle": args.handle})
    submissions = api.get("user.status", {
        "handle": args.handle, "from": 1, "count": 3
    })
    problemset = api.get("problemset.problems", {})

    print("API connectivity: OK")
    print("Profile keys:", sorted(profile.keys()))
    print("Rating-history rows:", len(rating))
    print("Submission rows:", len(submissions))
    if submissions:
        print("Submission keys:", sorted(submissions[0].keys()))
        print("Problem keys:", sorted((submissions[0].get("problem") or {}).keys()))
    print("Problem metadata rows:", len(problemset.get("problems", [])))
    print("Problem statistics rows:", len(problemset.get("problemStatistics", [])))

if __name__ == "__main__":
    main()
