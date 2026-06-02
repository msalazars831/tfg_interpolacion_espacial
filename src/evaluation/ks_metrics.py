import numpy as np

from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class KSMetrics:
    @staticmethod
    def _plot_pair(ax, rk_vals, cnn_vals, title=None, ks_stat=None, pval=None, bins=50, alpha=0.5):
        """Dibuja en `ax` los histogramas de `rk_vals` y `cnn_vals`.

        Parámetros:
        - `ax`: ejes matplotlib donde dibujar.
        - `title`: título a mostrar (string).
        - `ks_stat`, `pval`: valores opcionales para incluir en el título.
        """

        ax.hist(rk_vals, bins=bins, density=True, alpha=alpha, label='rk')
        ax.hist(cnn_vals, bins=bins, density=True, alpha=alpha, label='cnn')
        if ks_stat is not None and pval is not None:
            ax.set_title(f"{title}\nKS={ks_stat:.3f}, p={pval:.3g}")
        else:
            ax.set_title(title or "")
        ax.set_xlabel('valor')
        ax.set_ylabel('densidad')
        ax.legend()

    @staticmethod
    def ks_original(rk_values, cnn_values, plot=False, bins=50, alpha=0.5):
        """KS sobre valores estandarizados.

        Acepta `rk_std`/`cnn_std` precomputados para evitar recálculos.
        Si no se proporcionan, los calcula a partir de `rk_values`/`cnn_values`.
        """

        """KS sobre valores originales.

        Si `plot=True` dibuja un histograma comparativo.
        """

        statistic, pvalue = ks_2samp(rk_values, cnn_values)

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            KSMetrics._plot_pair(ax, rk_values, cnn_values, title='Original', ks_stat=statistic, pval=pvalue, bins=bins, alpha=alpha)
            fig.tight_layout()
            plt.show()

        return {"ks_statistic": statistic, "pvalue": pvalue}

    @staticmethod
    def ks_standardized(rk_values=None, cnn_values=None, rk_std=None, cnn_std=None, plot=False, bins=50, alpha=0.5):
        """KS sobre valores estandarizados.

        Acepta `rk_std`/`cnn_std` precomputados para evitar recálculos.
        Si no se proporcionan, los calcula a partir de `rk_values`/`cnn_values`.
        Si `plot=True` dibuja el histograma correspondiente.
        """

        if rk_std is None:
            if rk_values is None:
                raise ValueError("Se requiere `rk_values` o `rk_std`")
            rk_std = StandardScaler().fit_transform(rk_values.reshape(-1, 1)).ravel()

        if cnn_std is None:
            if cnn_values is None:
                raise ValueError("Se requiere `cnn_values` o `cnn_std`")
            cnn_std = StandardScaler().fit_transform(cnn_values.reshape(-1, 1)).ravel()

        statistic, pvalue = ks_2samp(rk_std, cnn_std)

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            KSMetrics._plot_pair(ax, rk_std, cnn_std, title='Estandarizado', ks_stat=statistic, pval=pvalue, bins=bins, alpha=alpha)
            fig.tight_layout()
            plt.show()

        return {"ks_statistic": statistic, "pvalue": pvalue}
    @staticmethod
    def ks_normalized(rk_values=None, cnn_values=None, rk_norm=None, cnn_norm=None, plot=False, bins=50, alpha=0.5):
        """KS sobre valores normalizados Min-Max.

        Acepta `rk_norm`/`cnn_norm` precomputados para evitar recálculos.
        Si no se proporcionan, los calcula a partir de `rk_values`/`cnn_values`.
        """

        if rk_norm is None:
            if rk_values is None:
                raise ValueError("Se requiere `rk_values` o `rk_norm`")
            rk_norm = MinMaxScaler().fit_transform(rk_values.reshape(-1, 1)).ravel()

        if cnn_norm is None:
            if cnn_values is None:
                raise ValueError("Se requiere `cnn_values` o `cnn_norm`")
            cnn_norm = MinMaxScaler().fit_transform(cnn_values.reshape(-1, 1)).ravel()

        statistic, pvalue = ks_2samp(rk_norm, cnn_norm)

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            KSMetrics._plot_pair(ax, rk_norm, cnn_norm, title='Normalizado', ks_stat=statistic, pval=pvalue, bins=bins, alpha=alpha)
            fig.tight_layout()
            plt.show()

        return {"ks_statistic": statistic, "pvalue": pvalue}

    @staticmethod
    def compute_all(rk_values, cnn_values, plot=False, bins=50, alpha=0.5):
        """Compute KS statistics for original, standardized and normalized arrays.

        Parameters:
        - `rk_values`, `cnn_values`: 1D numpy arrays of predictions.
        - `plot` (bool): si True, dibuja tres histogramas comparativos
          (original, estandarizado, normalizado).
        - `bins`, `alpha`: parámetros visuales para los histogramas.

        Devuelve un diccionario con las tres entradas: `original`,
        `standardized` y `normalized`, cada una con `ks_statistic` y
        `pvalue` (misma estructura que antes).
        """


        # Calcular transformaciones una vez
        orig = KSMetrics.ks_original(rk_values, cnn_values, plot=plot)

        # rk_std = StandardScaler().fit_transform(rk_values.reshape(-1, 1)).ravel()
        # cnn_std = StandardScaler().fit_transform(cnn_values.reshape(-1, 1)).ravel()
        std = KSMetrics.ks_standardized(rk_values=rk_values, cnn_values=cnn_values, plot=plot)

        # rk_norm = MinMaxScaler().fit_transform(rk_values.reshape(-1, 1)).ravel()
        # cnn_norm = MinMaxScaler().fit_transform(cnn_values.reshape(-1, 1)).ravel()
        norm = KSMetrics.ks_normalized(rk_values=rk_values, cnn_values=cnn_values, plot=plot)

        # if plot:
        #     fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        #     KSMetrics._plot_pair(axes[0], rk_values, cnn_values, title='Original', ks_stat=orig['ks_statistic'], pval=orig['pvalue'], bins=bins, alpha=alpha)
        #     KSMetrics._plot_pair(axes[1], rk_std, cnn_std, title='Estandarizado', ks_stat=std['ks_statistic'], pval=std['pvalue'], bins=bins, alpha=alpha)
        #     KSMetrics._plot_pair(axes[2], rk_norm, cnn_norm, title='Normalizado', ks_stat=norm['ks_statistic'], pval=norm['pvalue'], bins=bins, alpha=alpha)
        #     for ax in axes:
        #         ax.set_xlabel('valor')
        #         ax.set_ylabel('densidad')
        #     fig.tight_layout()
        #     plt.show()

        return {"original": orig, "standardized": std, "normalized": norm}