FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO noetic
ENV TZ=Asia/Taipei
ARG DEBIAN_FRONTEND=noninteractive
ARG APT_DPDS=apt_packages.txt
ARG PY_DPDS=requirements.txt

WORKDIR /tmp

USER root

# install apt dependencies
RUN apt update
COPY ./${APT_PKGS} ./
RUN xargs apt install \
    --yes \
    --no-install-recommends \
    < ${APT_DPDS}

# install python dependencies
COPY ./${PY_DPDS} ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install \
    --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --requirement ${PY_DPDS}

# Clean up
RUN apt autoremove --purge \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# setup entrypoint
# COPY ./entrypoint.sh /
# ENTRYPOINT ["/entrypoint.sh"]

WORKDIR /app

CMD ["bash"]