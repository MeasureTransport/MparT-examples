FROM quay.io/measuretransport/mpart_base:latest

USER root 
WORKDIR /

COPY .docker/environment.yml .

SHELL ["conda", "run", "-n", "mpart", "/bin/bash", "-c"]

RUN --mount=type=cache,target=/opt/conda/pkgs conda env update -n mpart -f environment.yml && \
    conda clean -afy && \
    rm environment.yml

WORKDIR /home/${NB_USER}
USER $NB_USER

COPY --chown=$NB_USER examples examples

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib \
    PYTHONPATH=$PYTHONPATH:/usr/local/python \
    OMP_PROC_BIND=spread \
    OMP_PLACES=threads \
    KOKKOS_NUM_THREADS=2

RUN cd $HOME/examples/python && \
    for i in *.py;do jupytext --set-formats ipynb,py $i; done

RUN cd $HOME/examples/python && \
    for i in *.ipynb;do jupyter nbconvert --ExecutePreprocessor.timeout=3600 --to notebook --execute $i; done

EXPOSE  8888
