import base64
import gzip
import io
import logging
import os
import glob
import time
from copy import deepcopy
from datetime import datetime
from typing import Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.visualization import (
    AsymmetricPercentileInterval,
    ImageNormalize,
    LinearStretch,
    LogStretch,
)
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from colorama import init as colorama_init
from colorama import Fore, Style

#import threading
#from threading import Thread
import multiprocessing as mp

logger = logging.getLogger(__name__)
# Initialize for printing color text
colorama_init()

DEFAULT_TIMEOUT = 5  # seconds


class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        super().__init__(*args, **kwargs)

    def send(self, request, *args, **kwargs):
        try:
            timeout = kwargs.get("timeout")
            if timeout is None:
                kwargs["timeout"] = self.timeout
            return super().send(request, *args, **kwargs)
        except AttributeError:
            kwargs["timeout"] = DEFAULT_TIMEOUT


class SendToFritz():
    base_key = "fritzsender"

    def __init__(
        self,
        base_name: str,
        group_ids: list,
        filter_id: int,
        instrument_id: int,
        stream_id: int,
        endpoint: str,
        update_thumbnails: bool = True,
        doSave: bool = False,
        protocol: str = "http",
    ):
        super().__init__()
        self.token = None
        self.group_ids = group_ids
        self.base_name = base_name
        self.filter_id = filter_id
        self.instrument_id = instrument_id
        self.origin = base_name  # used for sending updates to Fritz
        self.stream_id = stream_id
        self.endpoint = endpoint
        self.doSave = doSave
        self.protocol = protocol
        self.update_thumbnails = update_thumbnails

        # session to talk to SkyPortal/Fritz
        self.session = requests.Session()
        self.session_headers = {
            "Authorization": f"token {self._get_skyportal_token(self)}",
            "User-Agent": "DESIRT_BOT",
        }

        retries = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[405, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "POST", "PATCH"],
        )
        adapter = TimeoutHTTPAdapter(timeout=5, max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @staticmethod
    def _get_skyportal_token(self):
        if "fritz" in self.endpoint:
            token_skyportal = os.getenv("FRITZ_TOKEN")
        elif "desi" in self.endpoint:
            token_skyportal = os.getenv("NERSC_TOKEN")
        else:
            print(f"No token found for endpoint {self.endpoint}")

        if token_skyportal is None:
            err = (
                "No Fritz token specified. Run 'export FRITZ_TOKEN=<token>' "
                "and 'export NERSC_TOKEN=<token>'"
                "to set. The Fritz token will need to be specified manually "
                "for Fritz API queries."
            )
            logger.error(err)
            raise ValueError(err)

        return token_skyportal

    @staticmethod
    def _get_author_id():
        """Fritz author id is used in update calls.
        Can be found"""
        authid_fritz = os.getenv("FRITZ_AUTHID")

        if authid_fritz is None:
            err = (
                "No Fritz author id specified. Run 'export FRITZ_AUTHID=<id>' to set. "
                "Author id needs to be specified for updates sent by Fritz API queries."
            )
            logger.error(err)
            raise ValueError(err)

        return authid_fritz

    @staticmethod
    def read_input_df(df: pd.DataFrame):
        """Takes a DataFrame, which has multiple candidate
        and creates list of dictionaries, each dictionary
        representing a single candidate.

        Args:
            df (pandas.core.frame.DataFrame): dataframe of all candidates.

        Returns:
            (list[dict]): list of dictionaries, each a candidate.
        """
        all_candidates = []

        for i in range(0, len(df)):
            candidate = {}
            for key in df.keys():
                try:
                    if isinstance(df.iloc[i].get(key), (list, str)):
                        candidate[key] = df.iloc[i].get(key)
                    else:
                        # change to native python type
                        candidate[key] = df.iloc[i].get(key).item()
                except AttributeError:  # for IOBytes objs
                    try:
                        candidate[key] = df.iloc[i].get(key).getvalue()
                    except AttributeError:
                        # FIXME is this continue a problem?
                        continue
            all_candidates.append(candidate)

        return all_candidates

    def api(self, method: str, endpoint: str, data: Optional[Mapping] = None):
        """Make an API call to a SkyPortal instance

        headers = {'Authorization': f'token {self.token}'}
        response = requests.request(method, endpoint, json_dict=data,
                                    headers=headers)

        :param method:
        :param endpoint:
        :param data:
        :return:
        """
        method = method.lower()
        methods = {
            "head": self.session.head,
            "get": self.session.get,
            "post": self.session.post,
            "put": self.session.put,
            "patch": self.session.patch,
            "delete": self.session.delete,
        }

        if endpoint is None:
            raise ValueError("Endpoint not specified")
        if method not in ["head", "get", "post", "put", "patch", "delete"]:
            raise ValueError(f"Unsupported method: {method}")

        if method == "get":
            response = methods[method](
                f"{endpoint}",
                params=data,
                headers=self.session_headers,
            )
        else:
            response = methods[method](
                f"{endpoint}",
                json=data,
                headers=self.session_headers,
            )

        return response

    def alert_post_source(self, alert: dict,
                          group_ids: Optional[list] = None):
        """Add a new source to SkyPortal

        :param alert: dict of source info
        :param group_ids: list of group_ids to post source to, defaults to None
        """
        if group_ids is None:
            group_ids = self.group_ids
        data = {
            "ra": alert["ra"],
            "dec": alert["dec"],
            "id": alert["objectId"],
            "group_ids": group_ids,
            "origin": self.origin,
        }

        logger.debug(
            f"Saving {alert['objectId']} {alert['candid']} as a Source \
on SkyPortal"
        )
        response = self.api("POST", f"https://{self.endpoint}/api/sources", data)

        if response.json()["status"] == "success":
            logger.debug(
                f"Saved {alert['objectId']} {alert['candid']} as a Source \
on SkyPortal"
            )
        else:
            err = (
                f"Failed to save {alert['objectId']} {alert['candid']} "
                f"as a Source on SkyPortal"
            )
            logger.error(err)
            logger.error(response.json())

    def alert_post_candidate(self, alert):
        """
        Post a candidate on SkyPortal.
        Creates new candidate(s) (one per filter)
        """

        data = {
            "id": alert["objectId"],
            "ra": alert["ra"],
            "dec": alert["dec"],
            "filter_ids": [self.filter_id],
            "passing_alert_id": self.filter_id,
            "passed_at": Time(datetime.utcnow()).isot,
            "origin": "decamdrp",
        }
        logger.debug(
            f"Posting metadata of {alert['objectId']} {alert['candid']} to \
SkyPortal"
        )
        response = self.api("POST", f"https://{self.endpoint}/api/candidates", data)

        if response.json()["status"] == "success":
            logger.debug(
                f"Posted {alert['objectId']} {alert['candid']} metadata to \
SkyPortal"
            )
        else:
            logger.error(
                f"Failed to post {alert['objectId']} {alert['candid']} "
                f"metadata to SkyPortal"
            )
            logger.error(response.json())

    def make_thumbnail(self, alert, skyportal_type: str,
                       alert_packet_type: str):
        """
        Convert lossless FITS cutouts from ZTF-like alerts into PNGs.
        Make thumbnail for pushing to SkyPortal.

        :param alert: ZTF-like alert packet/dict
        :param skyportal_type: <new|ref|sub> thumbnail type expected
               by SkyPortal
        :param alert_packet_type: <Science|Template|Difference> survey naming
        :return:
        """
        alert = deepcopy(alert)
        cutout_data = alert[f"cutout{alert_packet_type}"]

        with gzip.open(io.BytesIO(cutout_data), "rb") as cutout:
            with fits.open(
                io.BytesIO(cutout.read()), ignore_missing_simple=True
            ) as hdu:
                image_data = hdu[0].data

        buff = io.BytesIO()
        plt.close("all")
        fig = plt.figure()
        fig.set_size_inches(4, 4, forward=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        # replace nans with median and flip orientation:
        img = np.array(image_data[::-1, ::-1])
        # replace dubiously large values
        xl = np.greater(np.abs(img), 1e20, where=~np.isnan(img))
        if img[xl].any():
            img[xl] = np.nan
        if np.isnan(img).any():
            median = float(np.nanmean(img.flatten()))
            img = np.nan_to_num(img, nan=median)

        norm = ImageNormalize(
            img,
            stretch=LinearStretch()
            if alert_packet_type == "Difference"
            else LogStretch(),
        )
        img_norm = norm(img)
        normalizer = AsymmetricPercentileInterval(
            lower_percentile=1, upper_percentile=100
        )

        vmin, vmax = alert[f"cutoutVmin{alert_packet_type}"], alert[f"cutoutVmax{alert_packet_type}"]
        # FIXME changed img_norm with img
        ax.imshow(img, cmap="bone", origin="lower", vmin=vmin, vmax=vmax)  # Lei comment: img_norm > img_norm[::-1, ::-1]
        plt.savefig(buff, dpi=42)

        buff.seek(0)
        plt.close("all")

        thumbnail_dict = {
            "obj_id": alert["objectId"],
            "data": base64.b64encode(buff.read()).decode("utf-8"),
            "ttype": skyportal_type,
        }

        return thumbnail_dict

    def alert_post_thumbnails(self, alert):
        """Post alert Science, Reference, and Subtraction thumbnails
        to SkyPortal

        :param alert: dict of source/candidate information
        :return:
        """
        for ttype, instrument_type in [
            ("new", "Science"),
            ("ref", "Template"),
            ("sub", "Difference"),
        ]:
            logger.debug(
                f"Making {instrument_type} thumbnail for {alert['objectId']} "
                f"{alert['candid']}",
            )
            thumb = self.make_thumbnail(alert, ttype, instrument_type)

            logger.debug(
                f"Posting {instrument_type} thumbnail for {alert['objectId']} "
                f"{alert['candid']} to SkyPortal",
            )
            response = self.api("POST",
                                f"https://{self.endpoint}/api/thumbnail", thumb)

            if response.json()["status"] == "success":
                logger.debug(
                    f"Posted {alert['objectId']} {alert['candid']} "
                    f"{instrument_type} cutout to SkyPortal"
                )
            else:
                logger.error(
                    f"Failed to post {alert['objectId']} {alert['candid']} "
                    f"{instrument_type} cutout to SkyPortal"
                )
                logger.error(response.json())

    def upload_thumbnail(self, alert):
        """Post new thumbnail to Fritz.

        NOTE: this is the original WINTER method for sending thumbnails,
        not full sized but higher contrast, similar to alert_make_thumbnail

        Format of thumbnail payload:
        { "obj_id": "string",  "data": "string",  "ttype": "string"}
        """
        fritz_to_cand = {"new": "SciBitIm", "ref": "RefBitIm",
                         "sub": "DiffBitIm"}

        for fritz_key, cand_key in fritz_to_cand.items():
            cutout = alert[cand_key]

            buffer = io.BytesIO()
            plt.figure(figsize=(3, 3))
            mean, median, std = sigma_clipped_stats(cutout)
            plt.imshow(
                cutout[::-1, ::-1],
                origin="lower",
                cmap="gray",
                vmin=mean - 1 * std,
                vmax=median + 3 * std,
            )
            plt.xticks([])
            plt.yticks([])

            plt.savefig(buffer, format="png")

            cutoutb64 = base64.b64encode(buffer.getvalue())
            cutoutb64_string = cutoutb64.decode("utf8")

            data_payload = {
                "obj_id": alert["objectId"],
                "data": cutoutb64_string,
                "ttype": fritz_key,
            }

            response = self.api(
                "POST", f"https://{self.endpoint}/api/thumbnail",
                data=data_payload
                )

            if response.json()["status"] == "success":
                logger.debug(
                    f"Posted {alert['objectId']} {alert['candid']} "
                    f"{cand_key} cutout to SkyPortal"
                )
            else:
                logger.error(
                    f"Failed to post {alert['objectId']} {alert['candid']} "
                    f"{cand_key} cutout to SkyPortal"
                )
                logger.error(response.json())

    def make_photometry(self, alert, jd_start: Optional[float] = None):
        """
        Make a de-duplicated pandas.DataFrame with photometry of
        alert['objectId']
        Modified from Kowalski (https://github.com/dmitryduev/kowalski)

        :param alert: candidate dictionary
        :param jd_start: date from which to start photometry from
        """
        alert = deepcopy(alert)
        top_level = [
            "schemavsn",
            "publisher",
            "objectId",
            "candid",
            "candidate",
            # FIXME removed PRV candidates here and below
            #"prv_candidates",
            "cutoutScience",
            "cutoutTemplate",
            "cutoutDifference",
        ]
        alert["candidate"] = {}

        # (keys having value in 3.)
        delete = [key for key in alert.keys() if key not in top_level]

        # delete the key/s
        for key in delete:
            alert["candidate"][key] = alert[key]
            del alert[key]

        alert["candidate"] = [alert["candidate"]]
        df_candidate = pd.DataFrame(alert["candidate"], index=[0])

        #df_prv_candidates = pd.DataFrame(alert["prv_candidates"])
        df_light_curve = df_candidate
        # FIXME reinstate PRV
        #df_light_curve = pd.concat(
        #    [df_candidate, df_prv_candidates], ignore_index=True, sort=False
        #)

        # note: WNTR (like PGIR) uses 2massj, which is not in sncosmo as of
        # 20210803, cspjs seems to be close/good enough as an approximation
        df_light_curve["filter"] = alert['candidate'][0]["filter"]

        df_light_curve["magsys"] = "ab"
        #df_light_curve["mjd"] = df_light_curve["jd"] - 2400000.5

        df_light_curve["mjd"] = df_light_curve["mjd"].astype(np.float64)
        df_light_curve["magpsf"] = df_light_curve["magpsf"].astype(np.float32)
        df_light_curve["sigmapsf"] = df_light_curve["sigmapsf"].astype(np.float32)

        df_light_curve = (
            df_light_curve.drop_duplicates(subset=["mjd", "magpsf"])
            .reset_index(drop=True)
            .sort_values(by=["mjd"])
        )

        # filter out bad data:
        mask_good_diffmaglim = df_light_curve["diffmaglim"] > 0
        df_light_curve = df_light_curve.loc[mask_good_diffmaglim]

        # convert from mag to flux

        # step 1: calculate the coefficient that determines whether the
        # flux should be negative or positive
        coeff = df_light_curve["isdiffpos"].apply(
            lambda x: 1.0 if x in [True, 1, "y", "Y", "t", "1"] else -1.0
        )

        # step 2: calculate the flux normalized to an arbitrary AB zeropoint of
        # 23.9 (results in flux in uJy)
        df_light_curve["flux"] = coeff * 10 ** (
            -0.4 * (df_light_curve["magpsf"] - 23.9)
        )

        # step 3: separate detections from non detections
        detected = np.isfinite(df_light_curve["magpsf"])
        undetected = ~detected

        # step 4: calculate the flux error
        df_light_curve["fluxerr"] = None  # initialize the column

        # step 4a: calculate fluxerr for detections using sigmapsf
        df_light_curve.loc[detected, "fluxerr"] = np.abs(
            df_light_curve.loc[detected, "sigmapsf"]
            * df_light_curve.loc[detected, "flux"]
            * np.log(10)
            / 2.5
        )

        # step 4b: calculate fluxerr for non detections using diffmaglim
        df_light_curve.loc[undetected, "fluxerr"] = (
            10 ** (-0.4 * (df_light_curve.loc[undetected, "diffmaglim"] - 23.9)) / 5.0
        )  # as diffmaglim is the 5-sigma depth

        # step 5: set the zeropoint and magnitude system
        df_light_curve["zp"] = 23.9
        df_light_curve["zpsys"] = "ab"

        # only "new" photometry requested?
        if jd_start is not None:
            w_after_jd = df_light_curve["jd"] > jd_start
            df_light_curve = df_light_curve.loc[w_after_jd]

        return df_light_curve

    def alert_put_photometry(self, alert, snr_thresh=3):
        """Send photometry to Fritz."""
        logger.debug(
            f"Making alert photometry of {alert['objectId']} {alert['candid']}"
        )

        df_photometry = self.make_photometry(alert)
        # post photometry
        photometry = {
            "obj_id": alert["objectId"],
            "stream_ids": [int(self.stream_id)],
            "group_ids": self.group_ids,
            "instrument_id": self.instrument_id,
            "mjd": df_photometry["mjd"].tolist(),
            #"flux": df_photometry["flux"].tolist(),
            #"fluxerr": df_photometry["fluxerr"].tolist(),
            "mag": df_photometry["magpsf"].tolist(),
            "magerr": df_photometry["sigmapsf"].tolist(),
            "snr": df_photometry["snr"].tolist(),
            #"zp": df_photometry["zp"].tolist(),
            "magsys": df_photometry["zpsys"].tolist(),
            "filter": df_photometry["filter"].tolist(),
            "limiting_mag_nsigma": df_photometry["limiting_mag_nsigma"].tolist(),
            "limiting_mag": df_photometry["diffmaglim"].tolist(),
            "ra": None,
            "dec": None,
        }
        
        try:
            if np.isnan(photometry["mag"][0]) == True or \
                    np.isnan(photometry["magerr"][0]):
                photometry["mag"][0] = None
                photometry["magerr"][0] = None
            # add check for SNR
            elif photometry["snr"][0] is not None:
                if photometry["snr"][0] < snr_thresh:
                    photometry["mag"][0] = None
                    photometry["magerr"][0] = None
            else:
                if photometry["ra"] is None and photometry["dec"] is None:
                    photometry["ra"] = df_photometry["ra"].tolist()
                    photometry["dec"] = df_photometry["dec"].tolist()
            if photometry["mag"][0] is None:
                photometry['snr'][0] = None
        except IndexError:
            print("WARNING: empty photometry")
            return
        # Remove the S/N key because not compliant with API
        photometry.pop('snr', None)

        if True:
        #if (len(photometry.get("flux", ())) > 0) or (
        #    len(photometry.get("fluxerr", ())) > 0
        #):
            logger.debug(
                f"Posting photometry of {alert['objectId']} {alert['candid']}, "
                f"stream_id={self.stream_id} to SkyPortal"
            )
            response = self.api(
                "PUT", f"https://{self.endpoint}/api/photometry", photometry
            )
            if response.json()["status"] == "success":
                logger.debug(
                    f"Posted {alert['objectId']} photometry \
stream_id={self.stream_id} "
                    f"to SkyPortal"
                )
            else:
                logger.error(
                    f"Failed to post {alert['objectId']} photometry "
                    f"stream_id={self.stream_id} to SkyPortal"
                )
                logger.error(response.json())

    def alert_post_annotation(self, alert):
        """Post an annotation. Works for both candidates and sources."""
        data = alert['annotation']
        payload = {"origin": self.origin, "data": data, "group_ids": self.group_ids}
        path = f'https://{self.endpoint}/api/sources/{str(alert["objectId"])}/annotations'
        response = self.api("POST", path, payload)

        if response.json()["status"] == "success":
            logger.debug(f"Posted {alert['objectId']} annotation to SkyPortal")
        else:
            logger.error(f"Failed to post {alert['objectId']} annotation to SkyPortal")
            logger.error(response.json())

    def alert_put_annotation(self, alert):
        """Retrieve an annotation to check if it exists already."""
        response = self.api(
            "GET",
            f'https://{self.endpoint}/api/sources/{str(alert["objectId"])}/annotations',
        )

        if response.json()["status"] == "success":
            logger.debug(f"Got {alert['objectId']} annotations from SkyPortal")
        else:
            logger.debug(
                f"Failed to get {alert['objectId']} annotations from SkyPortal"
            )
            logger.debug(response.json())
            return False

        existing_annotations = {
            annotation["origin"]: {
                "annotation_id": annotation["id"],
                "author_id": annotation["author_id"],
            }
            for annotation in response.json()["data"]
        }

        # no previous annotation exists on SkyPortal for this object? just post then
        if self.origin not in existing_annotations:
            self.alert_post_annotation(alert)
        # annotation from this(WNTR) origin exists
        else:
            # annotation data
            data = alert['annotation']
            new_annotation = {
                "author_id": existing_annotations[self.origin]["author_id"],
                "obj_id": alert["objectId"],
                "origin": self.origin,
                "data": data,
                "group_ids": self.group_ids,
            }

            logger.debug(
                f"Putting annotation for {alert['objectId']} {alert['candid']} "
                f"to SkyPortal",
            )
            response = self.api(
                "PUT",
                f"https://{self.endpoint}/api/sources/{alert['objectId']}"
                f"/annotations/{existing_annotations[self.origin]['annotation_id']}",
                new_annotation,
            )
            if response.json()["status"] == "success":
                logger.debug(
                    f"Posted updated {alert['objectId']} annotation to SkyPortal"
                )
            else:
                logger.error(
                    f"Failed to post updated {alert['objectId']} annotation "
                    f"to SkyPortal"
                )
                logger.error(response.json())

        return None

    def alert_skyportal_manager(self, alert):
        """Posts alerts to SkyPortal if criteria is met

        :param alert: _description_
        :type alert: _type_
        """
        # check if candidate exists in SkyPortal
        logger.debug(f"Checking if {alert['objectId']} is candidate in SkyPortal")
        # FIXME
        print("DEBUGGING")
        print(f"https://{self.endpoint}/api/candidates/{alert['objectId']}")
        response = self.api(
            "HEAD", f"https://{self.endpoint}/api/candidates/{alert['objectId']}"
        )
        is_candidate = response.status_code == 200
        logger.debug(
            f"{alert['objectId']} {'is' if is_candidate else 'is not'} "
            f"candidate in SkyPortal"
        )

        # check if source exists in SkyPortal
        logger.debug(f"Checking if {alert['objectId']} is source in SkyPortal")
        response = self.api(
            "HEAD", f"https://{self.endpoint}/api/sources/{alert['objectId']}"
        )
        is_source = response.status_code == 200
        logger.debug(
            f"{alert['objectId']} {'is' if is_source else 'is not'} source in SkyPortal"
        )
        # object does not exist in SkyPortal: neither cand nor source
        if (not is_candidate) and (not is_source):
            # post candidate
            try:
                self.alert_post_candidate(alert)
            except:
                print("Problem with posting the alert")
            # post annotations
            if alert['annotation'] is not None:
                try:
                    self.alert_post_annotation(alert)
                except:
                    print("Problem with posting the annotations")

            # post full light curve
            try:
                self.alert_put_photometry(alert)
            except:
                print("Problem with posting the photometry")
            # post thumbnails
            try:
                self.alert_post_thumbnails(alert)
            except:
                print("Problem with posting the thumbnails")

            # TODO autosave stuff, necessary?

        # obj already exists in SkyPortal
        else:
            # TODO passed_filters logic

            # post candidate with new filter ids
            self.alert_post_candidate(alert)

            if alert['annotation'] is not None:
                # put (*not* post) annotations
                self.alert_put_annotation(alert)

            # exists in SkyPortal & already saved as a source
            if is_source:
                # get info on the corresponding groups:
                logger.debug(
                    f"Getting source groups info on {alert['objectId']} from SkyPortal",
                )
                response = self.api(
                    "GET",
                    f"https://{self.endpoint}/api/sources/{alert['objectId']}/groups",
                )
                if response.json()["status"] == "success":
                    existing_groups = response.json()["data"]
                    existing_group_ids = [g["id"] for g in existing_groups]

                    for existing_gid in existing_group_ids:
                        if existing_gid in self.group_ids:
                            self.alert_post_source(alert, [str(existing_gid)])
                else:
                    logger.error(
                        f"Failed to get source groups info on {alert['objectId']}"
                    )
            else:  # exists in SkyPortal but NOT saved as a source
                if self.doSave is True:
                    self.alert_post_source(alert)

            # post alert photometry in single call to /api/photometry
            try:
                self.alert_put_photometry(alert)
            except:
                print("Problem adding the photometry")
            if self.update_thumbnails:
                try:
                    self.alert_post_thumbnails(alert)
                except:
                    print("Problem uploading the thumbnail")

        logger.debug(f'SendToFritz Manager complete for {alert["objectId"]}')

    def make_alert(self, cand_table, cand_annotations=None):
        t_0 = time.time()
        all_cands = self.read_input_df(cand_table)
        num_cands = len(all_cands)
        cand_annotated = []

        for cand in all_cands:
            print(f"Posting for {cand['candid']}, {cand['mjd']}, {cand['filter']}")

            if cand_annotations is not None and cand['objectId'] in cand_annotated:
                    cand['annotation'] = None
            else:
                cand['annotation'] = cand_annotations[cand['objectId']]
                cand_annotated.append(cand['objectId'])

            self.alert_skyportal_manager(cand)

        t_1 = time.time()
        logger.info(
            f"Took {(t_1 - t_0):.2f} seconds to Fritz process {num_cands} candidates."
        )

def makebitims(image: np.array):
    """
    make bit images of the cutouts for the marshal
    Args:
        image: input image cutout
    Returns:
        buf2: a gzipped fits file of the cutout image as
        a BytesIO object
    """
    # open buffer and store image in memory
    buf = io.BytesIO()
    buf2 = io.BytesIO()
    fits.writeto(buf, image)
    with gzip.open(buf2, "wb") as fz:
        fz.write(buf.getvalue())

    return buf2

def prepare_annotations(t_cands, t_alerts,
                        todel=['candid', 'skycoord_obj', 'ra', 'dec']):
    """
    Prepare a dataframe with the annotations information
    only for candidates with alerts to be posted

    Parameters
    ----------
    t_cands pd.DataFrame
        table with all the candidates summary info
    t_alerts pd.DataFrame
        table with all the alerts to be posted
    todel list
        list of keywords to remove from the annotations

    Returns
    -------
    dict_annotations dict
        dictionary with the annotation info (not posted already)
    """
    # object ID for alerts to be posted
    candids = set(t_alerts["objectId"])
    t_annotations = t_cands[t_cands['candid'].isin(candids)]
    t_annotations = t_annotations.fillna("NONE")
    # Create a dictionary for annotations
    dict_annotations = {i:t_annotations[t_annotations['candid'] == i].to_dict('records')[0] for i in candids}
    # Delete unnecessary keys
    if len(todel) > 0:
        for k in dict_annotations.keys():
            for i in np.arange(len(todel)):
                del dict_annotations[k][todel[i]]

    return dict_annotations


def prepare_alerts(t_cands, t_posted, path_cand=None, publisher=None,
                   endpoint=None, doSave=False):
    """
    Prepare a dataframe with all the alerts information

    Parameters
    ----------
    t_cands pd.DataFrame
        table with all the candidates summary info
    t_posted pd.DataFrame
        table with all the alerts already posted
    ...

    Returns
    -------
    cand_alerts pd.DataFrame object
        data frame with the info organized in alerts (not posted already)
    """
    # Create an empty table for the photometry
    photom = {"objectId": [],
              "mjd": [],
              "flux": [],
              "fluxerr": [],
              "magpsf": [],
              "sigmapsf": [],
              "diffmaglim": [],
              "limiting_mag_nsigma": [],
              "snr": [],
              "isdiffpos": [],
              "zp": [],
              "filter": [],
              "magsys": [],
              "zpsys": [],
              "cutoutScience": [],
              "cutoutTemplate": [],
              "cutoutDifference": [],
              "cutoutVminScience": [],
              "cutoutVmaxScience": [],
              "cutoutVminTemplate": [],
              "cutoutVmaxTemplate": [],
              "cutoutVminDifference": [],
              "cutoutVmaxDifference": [],
              "schemavsn": [],
              "publisher": []
}

    # Create a copy of the structure of the dataframe
    out_df = pd.DataFrame(data=None, columns=t_cands.columns)

    for idx, l in t_cands.iterrows():
        candname = l["candid"]
        cand_filenames = glob.glob(f"{path_cand}/*{candname}*fits")
        for cand_filename in cand_filenames:
            c_all = fits.open(cand_filename)[1].data
            # Check that the candidate has not been already posted
            tuples = [(candname, mjd, endpoint) for mjd in c_all["MJD_OBS"]]
            t_posted_tuples = [(l["objid"], l['mjd'], l['endpoint'])
                               for i,l in t_posted.iterrows()]
            idx_good = []
            for i, t in zip(np.arange(len(tuples)), tuples):
                # exclided posted candidates unless --doSave is given
                if not t in t_posted_tuples or doSave is True:
                    idx_good.append(i)
            mask = [True if i in idx_good else False
                    for i in np.arange(len(c_all))]
            # Check that there is at least one
            if len(idx_good) == 0:
                continue
            # Select only those to post
            # FIXME reinstate
            c = c_all[0:2]
            ####c = c_all[mask]
            photom["objectId"] += [candname] * len(c)
            photom["mjd"] += c["MJD_OBS"].tolist()
            photom["zp"] += c["ZP_FPHOT"].tolist()
            photom["filter"] += ['des'+filt.lower() for filt in c["FILTER"]]
            photom["magpsf"] += c["MAG_FPHOT"].tolist()
            photom["sigmapsf"] += c["MAGERR_FPHOT"].tolist()
            photom["diffmaglim"] += c["LIM_MAG5"].tolist()
            photom["limiting_mag_nsigma"] += [5] * len(c)
            photom["snr"] += c["SNR_FPHOT"].tolist()
            photom["isdiffpos"] += [1] * len(c)
            photom["magsys"] += ["ab"] * len(c)
            photom["zpsys"] += ["ab"] * len(c)
            photom["cutoutScience"] += [makebitims(x) for x in c['PixA_THUMB_SCI']]
            photom["cutoutTemplate"] += [makebitims(x) for x in c['PixA_THUMB_TEMP']]
            photom["cutoutDifference"] += [makebitims(x) for x in c['PixA_THUMB_DIFF']]
            photom["cutoutVminScience"] += c['ZMIN_SCI'].tolist()
            photom["cutoutVmaxScience"] += c['ZMAX_SCI'].tolist()
            photom["cutoutVminTemplate"] += c['ZMIN_TEMP'].tolist()
            photom["cutoutVmaxTemplate"] += c['ZMAX_TEMP'].tolist()
            photom["cutoutVminDifference"] += c['ZMIN_DIFF'].tolist()
            photom["cutoutVmaxDifference"] += c['ZMAX_DIFF'].tolist()
            photom["schemavsn"] += ["test"] * len(c)
            photom["publisher"] += [publisher] * len(c)

            # Add rows to the output dataframe
            for i in np.arange(len(c)):
                out_df = out_df._append(t_cands.iloc[idx], ignore_index=True)

        # Calculate flux and flux error
        mags = np.array(photom["magpsf"])
        errs = np.array(photom["sigmapsf"])
        zp = np.array(photom["zp"])
        photom["flux"] = 10**(-1. * (mags - zp)/2.5)
        photom["fluxerr"] = np.abs(10**(-1. * (mags + errs - zp)/2.5) -
                                   10**(-1. * (mags - zp)/2.5))


    # Update the table
    for k in photom.keys():
        out_df[k] = photom[k]

    return out_df

def prepare_and_send(cand_df, stf, posted_file, posted, path_cand, publisher,
                     endpoint, doSave):
    """
    Take a dataframe as input, prepares the alert and posts it.
    Operations grouped in a function to better handle multiprocessing

    Parameters
    ----------
    cand_df pandas data frame
        data frame with one or multiple candidates
    posted pandas data frame
        candidates already posted
    posted_file str
        filename of posted candidates
    """
    cand_alerts = prepare_alerts(cand_df, posted,
                                 path_cand=path_cand,
                                 publisher=publisher,
                                 endpoint=endpoint, doSave=doSave)
    # Prepare global info for annotations 
    cand_annotations = prepare_annotations(cand_df, cand_alerts)

    # For debugging
    # FIXME
    #cand_alerts = cand_alerts[cand_alerts['candid'] == "A202302230957344p005754"]
    #cand_alerts_short = cand_alerts.iloc[0:20]
    stf.make_alert(cand_alerts, cand_annotations=cand_annotations)
    # Write the reported alerts
    # FIXME risk of writing multiple times
    with open(posted_file, "a") as out:
        for idx, c in cand_alerts.iterrows():
            out.write(f"{c['candid']},{c['mjd']},{endpoint}\n")


def sendToSkyPortal(path_base, programid, dateobs, endpoint,
                    summaryfile="DECam_Candidate_Summary.csv",
                    base_name="basename", publisher="iandreoni",
                    timeGap=10, doSave=False, doParallel=True,
                    names=None, onePush=False, group=None):
    # Build the paths
    path_cand = f"{path_base}/{programid}/{dateobs}/NORMAL/Candidates".replace("//", "/")
    cands_file = f"{path_base}/{programid}/{dateobs}/NORMAL/{summaryfile}".replace("//", "/")
    posted_file = f"{path_base}/{programid}/{dateobs}/NORMAL/Alerts_posted_{endpoint}.csv".replace("//", "/")

    # Check that the path exists:
    if os.path.exists(path_cand) is True:
        pass
    else:
        print("The path does not exist!")
        print(path_cand)
        print("Check the input values")
        print(f"pathBase: {path_base}")
        print(f"program: {path_base}")
        print(f"date: {dateobs}")
        exit()
    # Check that the summary file exists
    if os.path.exists(cands_file) is True:
        pass
    else:
        print(f"Candidates summary file not found at {cands_file}")
        print("Exiting...")
        exit()
    # Check that the posted file exists
    if os.path.exists(posted_file) is True:
        pass
    else:
        try:
            with open(posted_file, "w") as f:
                f.write("objid,mjd,endpoint\n")
        except Exception as e:
            print(f"PROBLEMS WRITING THE FILE {posted_file}: ", e)
            exit()

    # Endpoint: desi-skyportal.lbl.gov, fritz.science, preview.fritz.science
    if endpoint == "fritz.science":
        # https://fritz.science/api/groups
        if group != 1548:
            print(f"WARNING: group {group} not DECAM-GW")
        group_ids = [group]
        # https://fritz.science/api/filters
        filter_id = 1157
        # https://fritz.science/api/instrument
        instrument_id = 54
        # https://fritz.science/api/streams
        stream_id = 1006
    elif endpoint == "desi-skyportal.lbl.gov":
        # https://fritz.science/api/groups
        if group != 119:
            print(f"WARNING: group {group} not DECAM-GW")
        group_ids = [group]
        # https://fritz.science/api/filters
        filter_id = 47
        # https://fritz.science/api/instrument
        instrument_id = 8
        # https://fritz.science/api/streams
        stream_id = 42
    stf = SendToFritz(base_name, group_ids, filter_id, instrument_id,
                      stream_id, endpoint, doSave=doSave)
    # Start the listener
    starttime = time.time()
    # Time gap from minutes to seconds
    timeGap_s = timeGap * 60
    doRepeat = True
    while doRepeat is True:
        cands = glob.glob(f"{path_cand}/*fits")
        print("STARTING ITERATION")
        print(f"There are {len(cands)} candidates in {path_cand}")
        if len(cands) > 0:
            # Read the candidates summary file
            all_cands = pd.read_csv(cands_file)
            # Rename columns to better interact with the other functions
            all_cands = all_cands.rename(columns={"objid": "candid",
                                  "ra_obj": "ra",
                                  "dec_obj": "dec"})
            # Read the file with alerts already posted
            posted = pd.read_csv(posted_file)
            # Check if a specific candidate of interest is there
            if names is not None:
                names = [n if n in all_cands['candid'].values
                         else print(f"{Fore.RED}Candidate file missing for {n} \
The object will NOT be pushed to {endpoint} {Style.RESET_ALL}")
                         for n in names
                         ]
                # Any object left after the selection?
                if len([n for n in names if not (n is None)]) == 0:
                    print("No input candidates to push, exiting...")
                    exit()
                else:
                    # Table only for sources present in the candidate files
                    idx_select = [i for i in np.arange(len(all_cands)) if
                                  all_cands['candid'][i] in names]
                    all_cands = all_cands.iloc[idx_select]
                    all_cands.index = np.arange(len(all_cands))
            # Iterate over candidates
            if doParallel is False:
                prepare_and_send(all_cands, stf, posted_file,
                                 posted,
                                 path_cand,
                                 publisher,
                                 endpoint,
                                 doSave)

            else:
                # Prepare and send candidates in parallel
                ncpu = 4
                pool = mp.Pool(processes=ncpu)
                process_list = []
                # Individual dataframes for each candidate
                # FIXME apply to DECam, this is from winter/mirar
                #workers = []
                #for _ in range(ncpu):
                #    # Set up a worker thread to process database load
                #    worker = Thread(
                #        target=self.apply_to_batch, args=(watchdog_queue, cache_id)
                #        )
                #    worker.daemon = True
                #    worker.start()
                #    workers.append(worker)

                for candid in all_cands['candid']:
                    cand_df = all_cands[all_cands['candid'] == candid].reset_index()
                    print("candidate:")
                    print(cand_df)
                    process_list.append(pool.apply_async(prepare_and_send,
                                        args=(cand_df, stf,
                                              posted,
                                              path_cand,
                                              publisher,
                                              endpoint,
                                              doSave,
                                              )))
                     #p = pool.apply(prepare_and_send,
                # Add a progress bar
                #process_list = tqdm(process_list)
                results = [p.get() for p in process_list]
                pool.close()

        print(f"Done in {(time.time() - starttime)}s")
        if onePush is True:
            doRepeat = False
        else:
            # Remove the Time taken by code to execute
            time.sleep(timeGap_s - ((time.time() - starttime) % timeGap_s))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Send candidates to SkyPortal')
    parser.add_argument('--program', dest='program', type=str,
                        required=True,
                        help='<Required> DECam program ID',
                        default="")
    parser.add_argument('-d', '--date', dest='date', type=str,
                        required=True,
                        help='<Required> Date of interest (e.g.: 2023-02-22)',
                        default="2023-02-22")
    parser.add_argument('-ep', '--endpoint', dest='endpoint', type=str,
                        required=False,
                        help='Skyportal endpoint (desi-skyportal.lbl.gov \
or fritz.science) - default desi at NERSC',
                        default="desi-skyportal.lbl.gov")
    parser.add_argument('--pathBase', dest='path_base', type=str,
                        required=False,
                        help='Base path (where the pipeline runs)',
                        default="/Users/igor/Software/decam-dps")
    parser.add_argument('--publisher', dest='publisher', type=str,
                        required=False,
                        help='Publisher on SkyPortal',
                        default="iandreoni")
    parser.add_argument('--group', dest='group', type=int,
                        required=False,
                        help='SkyPortal group',
                        default=119)
    parser.add_argument('-tg', '--timeGap', dest='timeGap', type=float,
                        required=False,
                        help='Time gap (in minutes) between executions',
                        default=5.)
    parser.add_argument('--doSave', action="store_true", default=False,
                        required=False,
                        help='Save the object as a source automatically')
    parser.add_argument('-n','--name', nargs='+', dest='names',
                        help='List of candidate names',
                        default=None, required=False)
    parser.add_argument('--onePush', action="store_true", default=False,
                        required=False,
                        help='Do not repeat the pushing every timeGap minutes')
    parser.add_argument('--doParallel', action="store_true", default=False,
                        required=False,
                        help='Use multithreading parallelization')
    args = parser.parse_args()

    # Run the main code
    sendToSkyPortal(args.path_base, args.program, args.date, args.endpoint,
                    base_name="sfft_pipe", publisher=args.publisher,
                    timeGap=args.timeGap, doSave=args.doSave,
                    doParallel=args.doParallel, names=args.names,
                    onePush=args.onePush, group=args.group)
