# from ..simulate import SourceBase, Processor
# import perceval as pcvl
# import numpy as np
#
#
# # Displays some results for Source and Processor zpg
# def display_results(obj):
#     if isinstance(obj, Processor):
#         print('{:}:  {:}'.format(*['Outcome', 'Probability']))
#         for key, value in obj.probs(chop=True).items():
#             if value >= 10 ** (-obj.precision + 1):
#                 print('{:}:  {:.{p}f}'.format(*[key, value], p=obj.precision - 2))
#         print('')
#     elif isinstance(obj, SourceBase):
#         print('')
#         print('{:}:  {:}'.format(
#             *['Figure of Merit', 'Measurement']))
#         for key, value in obj.quality.items():
#             if isinstance(value, float):
#                 print('{:}:  {:.{p}f}'.format(*[key, value], p=obj.precision - 2))
#         if any(n[:-2] == 'pn' for n in obj.quality.keys()):
#             print('')
#             print('{:}:  {:}'.format(
#                 *['Photon number', 'Probability']))
#             for md in range(0, obj.modes):
#                 if 'pn ' + str(md) in obj.quality.keys():
#                     maxp = round(min(-np.log10(list(obj.quality['pn ' + str(md)].values())[-1] + 10 ** -20), obj.precision - 2))
#                     print('{:}:  {:}'.format(*['Mode', md]))
#                     for n, pn in list(
#                             zip(*[range(0, len(obj.quality['pn ' + str(md)])),
#                                   obj.quality['pn ' + str(md)].values()])):
#                         print('{:}:  {:.{p}f}'.format(*[n, pn], p=maxp))
#         elif 'pn' in obj.quality.keys():
#             maxp = round(min(-np.log10(list(obj.quality['pn'].values())[-1] + 10 ** -20), obj.precision - 2))
#             print('')
#             print('{:}:  {:}'.format(
#                 *['Photon number', 'Probability']))
#             for n, pn in list(
#                     zip(*[range(0, len(obj.quality['pn'])), obj.quality['pn'].values()])):
#                 print('{:}:  {:.{p}f}'.format(*[n, pn], p=maxp))
#
#
# def tvd(p: Processor, istate: list = None):
#     istate = [1 if inpt.is_assigned else 0 for inpt in p.inputs] if istate is None else istate
#     if hasattr(p, 'circuit'):
#         if p.probs:
#             pideal = pcvl.Processor('SLOS', p.circuit)
#             pideal.with_input(pcvl.BasicState(istate))
#             pcvl_probs = pideal.probs()['results']
#             tvd = 0
#             for k, v in pcvl_probs.items():
#                 key = ''.join([str(i) for i in k])
#                 tvd += abs(v - (p.probs()[key] if key in p.probs().keys() else 0)) / 2
#             return tvd
#         else:
#             assert False, "No probabilities simulated."
#     else:
#         assert False, "Processor has no circuit defined."
