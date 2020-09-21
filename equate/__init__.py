import re
import os
from collections import Counter

from py2store import DirReader, filt_iter, LocalTextStore


def features_of_site_packages(site_packages_dir=None):
    """Iterator of package features extracted from site_packages_dir.
    If site_packages_dir is not given, will take the first directory found by `site.getsitepackages()`

    To get a dataframe containing all the packages and features found:

    >>> import pandas as pd
    >>> df = pd.DataFrame(list(map(dict, features_of_site_packages())))
    """
    if site_packages_dir is None:
        # If not given explicitly, find it within the current env, using site package
        import site
        site_packages_dir = next(iter(site.getsitepackages()), None)

    pkgs_info_dirs = filt_iter(DirReader(site_packages_dir),
                               filt=lambda x: x.endswith('info/'))
    for pkg_info_dir_path in pkgs_info_dirs:
        yield dist_info_features(pkg_info_dir_path)


path_sep = os.path.sep

first_word_re = re.compile('\w+')
first = lambda g, dflt=None: next(iter(g), dflt)


def dist_info_features(dist_info_dir):
    if dist_info_dir.endswith(path_sep):
        dist_info_dir = dist_info_dir[:-1]
    dist_info_dirname = os.path.basename(dist_info_dir)
    dist_name = first_word_re.match(dist_info_dirname).group(0)
    yield ('dist_name', dist_name)

    ss = LocalTextStore(dist_info_dir)

    if 'RECORD' in ss:
        lines = ss['RECORD'].split('\n')
        top_dirnames = (first(line.split(path_sep), '')
                        for line in lines if not line.endswith(dist_info_dirname))
        yield ('most_frequent_record_dirname', first(Counter(top_dirnames).most_common(), [None, None])[0])

    if 'top_level.txt' in ss:
        yield ('first_line_of_top_level_txt', first(ss['top_level.txt'].split('\n')))

    if 'INSTALLER' in ss:
        yield ('installer', first(ss['INSTALLER'].split('\n')))

    if 'METADATA' in ss:
        ww = dict(filter(lambda w: len(w) == 2,
                         (re.split(': ', x, maxsplit=1) for x in ss['METADATA'].split('\n'))))
        yield ('metadata_name', ww.get('Name', None))

        m = re.compile('(?<=pypi.org/project/)[^/]+').search(ww.get('Download-URL', ''))
        yield ('download_url_name', m and m.group(0))


#

def file_peeps(store, peep_size=100):
    for kk, vv in store.items():
        yield kk, vv[:peep_size]


def file_peeps_print(store, peep_size=100):
    for kk, vv in file_peeps(store, peep_size):
        print(f"---- {kk} ----")
        print(f"{vv}")
        if len(vv) > peep_size:
            print('...')
        print(f"{'-' * (len(kk) + 4 + 4 + 2)}")
        print("")


def print_n_null_elements_in_each_column_containing_at_least_one(df):
    for c in df.columns:
        n = len(df) - sum(df[c].notnull())
        if n > 0:
            print(f"{c}:\t{n} null values")
