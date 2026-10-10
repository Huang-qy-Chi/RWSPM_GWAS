# Third-party source notices

The `cball_ext/` directory contains Ball-related C source and local wrappers.
The original author and copyright comments have been retained. In particular,
`utilities.c` declares copyright 2017 ChengFeng Liu and Jin Zhu and permission
under GNU GPL version 3 or, at the recipient's option, any later version.

The GNU GPL v3 text is included in [COPYING.GPL-3.0.txt](COPYING.GPL-3.0.txt),
obtained from <https://www.gnu.org/licenses/gpl-3.0.txt>.

This snapshot does not assign a new blanket license to the project's other
original code. The included GPL text documents the existing third-party notice;
it is not a claim that every file was authored or licensed by the Ball authors.
No MIT, Apache, or other replacement license has been added.

Background:

- Pan et al., *Ball Covariance: A Generic Measure of Dependence in Banach Space*:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC7720858/>.
- The C functions `bdd_matrix_bias`, distance/rank utilities, and median helpers
  are retained with their source comments; `snp_bcov.c` supplies the project's
  SNP-specialised statistic and rank interface.
