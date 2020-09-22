import re
import os
from collections import Counter
import site
from typing import Any, Mapping

from py2store import DirReader, filt_iter, LocalTextStore

DFLT_SITE_PKG_DIR = next(iter(site.getsitepackages()), None)


def site_packages_info_df(site_packages_dir=DFLT_SITE_PKG_DIR):
    """Get a pandas.DataFrame of package features extracted from site_packages_dir.
    """
    import pandas as pd
    df = pd.DataFrame(list(map(dict, features_of_site_packages(site_packages_dir))))
    df._site_packages_dir = site_packages_dir
    return df


def features_of_site_packages(site_packages_dir=DFLT_SITE_PKG_DIR):
    """Iterator of package features extracted from site_packages_dir.

    To get a dataframe containing all the packages and features found:

    >>> import pandas as pd
    >>> df = pd.DataFrame(list(map(dict, features_of_site_packages())))
    """

    pkgs_info_dirs = site_dir_reader(site_packages_dir, filt=lambda x: x.endswith('info/'))
    for pkg_info_dir_path in pkgs_info_dirs:
        yield dist_info_features(pkg_info_dir_path)


def site_dir_reader(site_packages_dir=DFLT_SITE_PKG_DIR, filt=None):
    """Store of directories of site_packages_dir.
    If site_packages_dir is not given, will take the first directory found by `site.getsitepackages()`
    """

    if filt:
        store = filt_iter(DirReader(site_packages_dir), filt=filt)
    else:
        store = DirReader(site_packages_dir)
    store._madeby = dict(
        func=site_dir_reader,
        kwargs=dict(site_packages_dir=site_packages_dir, filt=filt)
    )
    return store


path_sep = os.path.sep

first_word_re = re.compile('\w+')
first = lambda g, dflt=None: next(iter(g), dflt)


def last(g, dflt=None):
    x = dflt
    for x in g:
        pass
    return x


class NotSpecified:
    pass


def key_chain(d: Mapping, /, *keys, dflt: Any = NotSpecified):
    """Get a value from a mapping, from a list of possible key names. First one found wins.
    You know collections.Chain? That's to look for a single key in a chain of mappings.
    This is to find

    >>> d = {'a': 1, 'b': 2, 'c': 3}
    >>> key_chain(d, 'x', 'c', 'b')
    3
    >>> assert key_chain(d, 'x', 'y', dflt=None) == None
    >>> key_chain(d, 'x', 'y')
    Traceback (most recent call last):
      ...
    KeyError: "Couldn't find any of these keys: x, y"

    """
    for k in keys:
        if k in d:
            return d[k]
    if dflt is not NotSpecified:
        return dflt
    else:
        raise KeyError(f"Couldn't find any of these keys: {', '.join(keys)}")


def dist_info_features(dist_info_dir):
    if dist_info_dir.endswith(path_sep):
        dist_info_dir = dist_info_dir[:-1]

    dist_info_dirname = os.path.basename(dist_info_dir)
    yield ('dist_info_dirname',
           dist_info_dirname)
    yield ('info_kind',
           last(dist_info_dirname.split('.')))

    dist_name = first_word_re.match(dist_info_dirname).group(0)
    yield ('dist_name',
           dist_name)

    ss = LocalTextStore(dist_info_dir)

    def record_lines(records_txt):
        for line in records_txt.split('\n'):
            if not line.startswith(dist_info_dirname):
                if line.startswith('__pycache__' + path_sep):
                    line = line[len('__pycache__' + path_sep):]
                    yield line[:line.index('.')]
                else:
                    if line.startswith('..' + path_sep):
                        line = line[3:]  # discard a ../ prefix
                    yield line

    record_txt = key_chain(ss, 'RECORD', 'installed-files.txt', dflt=None)
    if record_txt is not None:
        top_dirnames = (first(line.split(path_sep), '') for line in record_lines(record_txt))
        yield ('most_frequent_record_dirname',
               first(Counter(top_dirnames).most_common(), [None, None])[0])

    if 'top_level.txt' in ss:
        yield ('first_line_of_top_level_txt',
               first(ss['top_level.txt'].split('\n')))

    if 'INSTALLER' in ss:
        yield ('installer',
               first(ss['INSTALLER'].split('\n')))

    metadata = key_chain(ss, 'METADATA', 'PKG-INFO', dflt=None)
    if metadata is not None:
        ww = dict(filter(lambda w: len(w) == 2,
                         (re.split(': ', x, maxsplit=1) for x in metadata.split('\n'))))
        yield ('metadata_name',
               ww.get('Name', None))

        def gen():
            for v in ww.values():
                m = re.compile('(?<=pypi.org/project/)[^/]+').search(v)
                if m is not None:
                    yield m.group(0)

        yield ('pypi_url_name',
               next(gen(), None))
        # m = re.compile('(?<=pypi.org/project/)[^/]+').search(ww.get('Download-URL', ''))
        # yield ('download_url_name', m and m.group(0))


# Diagnosis objects

class Lidx:
    """A df query class"""
    diagnosis_meths = ('no_nans', 'equal', 'dash_underscore_eq', ('equal', 'dash_underscore_eq'))

    def __init__(self,
                 df,
                 import_name='most_frequent_record_dirname',
                 pkg_name='metadata_name'):
        self.df = df.rename(columns={import_name: 'import_name', pkg_name: 'pkg_name'})

    @property
    def no_nans(self):
        return ~(self.df.import_name.isna() | self.df.pkg_name.isna())

    @property
    def equal(self):
        return self.df.import_name == self.df.pkg_name

    @property
    def dash_underscore_eq(self):
        underscored_pkg_names = [(isinstance(x, str) and x.replace('-', '_') or x) for x in self.df.pkg_name.values]
        return self.df.import_name == underscored_pkg_names

    def print_diagnosis(self, diagnosis_meths=None):
        if diagnosis_meths is None:
            diagnosis_meths = self.diagnosis_meths
        for meth in diagnosis_meths:
            if isinstance(meth, str):
                r = sum(getattr(self, meth))
            elif isinstance(meth, tuple):
                from functools import reduce
                r = sum(reduce(lambda x, y: x & y,
                               [getattr(self, m) for m in meth],
                               [True] * len(self.df)))
            else:
                raise TypeError(f"Don't know how to handle meth: {meth}")
            print(f"{meth}: {r}")


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
