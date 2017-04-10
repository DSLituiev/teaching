from sklearn.tree import export_graphviz
from IPython.display import Image
# install pydotplus (if not installed) for visualization of decision tree diagrams 
# !pip3 install pydotplus
import pip
import importlib

import re
def import_or_install(package):
    try:
        #equivalent of `import pydotplus`
        pkg = importlib.import_module(package)
#         import pydotplus
        print("imported %s" % package)
    except ImportError:
        pip.main(['install', package])
        pkg = importlib.import_module(package)
    return pkg

package="pydotplus"
pydotplus = import_or_install(package)

def _get_dot_(dtree, feature_names, flag_cols=[], class_names=["NO","YES"]):
    dot_data = export_graphviz(dtree, out_file=None,
                             feature_names= feature_names,
                             class_names=class_names,
                             filled=True, rounded=True,
                             impurity=False,
                             special_characters=True)

    pattern_txt = "label=<([a-zA-Z,_,0-9]+ .*)<br/>samples = (\d*)<br/>value = \[(\d*), (\d*)\]<br/>class = [a-zA-Z]*>"
    out_pattern = "label=<(\\2)<br/>NO: \\3 | YES: \\4<br/> <br/>\\1>"
    pattern = re.compile(pattern_txt, re.UNICODE)
    dot_data = pattern.sub(out_pattern, dot_data)

    pattern_txt = "label=<samples = (\d*)<br/>value = \[(\d*), (\d*)\]<br/>class = [a-zA-Z]*>"
    out_pattern = "label=<(\\1)<br/><b>NO:</b> \\2 | YES: \\3>"
    pattern = re.compile(pattern_txt, re.UNICODE)
    dot_data = pattern.sub(out_pattern, dot_data)

    for col in flag_cols:
        pattern_txt = "{}_([a-z_]+.*) &le; 0\.5".format(col,)
        out_pattern = "{} != \\1".format(col)
        # print(pattern_txt)
        pattern = re.compile(pattern_txt, re.UNICODE)
        dot_data = pattern.sub(out_pattern, dot_data)
    return dot_data


def _visualize_tree_(dot_data):
    graph = pydotplus.graph_from_dot_data(dot_data)
    return Image(graph.create_png())


def visualize_tree(dtree, feature_names, flag_cols=[], class_names=["NO","YES"]):
    dot_data = _get_dot_(dtree, feature_names, flag_cols=flag_cols, class_names=class_names)
    return _visualize_tree_(dot_data)
