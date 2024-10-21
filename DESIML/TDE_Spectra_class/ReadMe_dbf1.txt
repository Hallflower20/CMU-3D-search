================================================================================
Title: The Final Season Reimagined: 30 Tidal Disruption Events from the 
       ZTF-I Survey 
Authors: Hammerstein E., van Velzen S., Gezari S., Cenko S.B., Yao Y., Ward C.,
    Frederick S., Villanueva N., Somalwar J.J., Graham M.J., Kulkarni S.R., 
    Stern D., Andreoni I., Bellm E.C., Dekany R., Dhawan S., Drake A.J., 
    Fremling C., Gatkine P., Groom S.L., Ho A.Y.Q., Kasliwal M.M., 
    Karambelkar V., Kool E.C., Masci F.J., Medford M.S., Perley D.A., 
    Purdum J., van Roestel J., Sharma Y., Sollerman J., Taggart K., Yan L.
================================================================================
Description of contents: This tar.gz archive contains the data presented in 
    Figure 1 of the accepted version of the article listed above. These spectra
    were used for the spectral classifications of the ZTF-I TDE sample. 
    The specific data files provided in this archive (60) are given below:

    IAU        ZTF           MRT Version               Original Version
    ----       ----          ----                      ----
    AT2018lni  ZTF18actaqdw  AryaStark_DCT.mrt         AryaStark_DCT.ascii
    AT2019dsg  ZTF19aapreis  BranStark_DCT.mrt         BranStark_DCT.ascii
    AT2019ehz  ZTF19aarioci  Brienne_DCT.mrt           Brienne_DCT.ascii
    AT2019mha  ZTF19abhejal  Bronn_DBSP.mrt            Bronn_DBSP.ascii
    AT2018lna  ZTF19aabbnzo  CerseiLannister_GMOS.mrt  CerseiLannister_GMOS.ascii
    AT2018hyz  ZTF18acpdvos  GendryBaratheon_FTN.mrt   GendryBaratheon_FTN.ascii
    AT2020pj   ZTF20aabqihu  Gilly_DCT.mrt             Gilly_DCT.ascii
    AT2020opy  ZTF20abjwvae  HighSparrow_DCT.mrt       HighSparrow_DCT.ascii
    AT2020zso  ZTF20acqoiyt  Hodor_LRIS.mrt            Hodor_LRIS.ascii
    AT2019azh  ZTF17aaazdba  JaimeLannister_DCT.mrt    JaimeLannister_DCT.ascii
    AT2018bsi  ZTF18aahqkbt  JonSnow_DCT.mrt           JonSnow_DCT.ascii
    AT2018iih  ZTF18acaqdaa  JorahMormont_DCT.mrt      JorahMormont_DCT.ascii
    AT2020qhs  ZTF20abowque  Loras_DCT.mrt             Loras_DCT.ascii
    AT2019meg  ZTF19abhhjcc  MargaeryTyrell_DBSP.mrt   MargaeryTyrell_DBSP.ascii
    AT2019qiz  ZTF19abzrhgq  Melisandre_DCT.mrt        Melisandre_DCT.ascii
    AT2019teq  ZTF19accmaxo  Missandei_DCT.mrt         Missandei_DCT.ascii
    AT2018zr   ZTF18aabtxvd  NedStark_DCT.mrt          NedStark_DCT.ascii
    AT2020ysg  ZTF20abnorit  Osha_DCT.mrt              Osha_DCT.ascii
    AT2019cho  ZTF19aakiwze  PetyrBalish_DCT.mrt       PetyrBalish_DCT.ascii
    AT2020ocn  ZTF18aakelin  Podrick_DBSP.mrt          Podrick_DBSP.ascii
    AT2020mot  ZTF20abfcszi  Pycelle_DCT.mrt           Pycelle_DCT.ascii
    AT2019lwu  ZTF19abidbya  RobbStark_DCT.mrt         RobbStark_DCT.ascii
    AT2020wey  ZTF20acitpfz  Roose_FTN.mrt             Roose_FTN.ascii
    AT2018jbv  ZTF18acnbpmd  SamwellTarly_DCT.mrt      SamwellTarly_DCT.ascii
    AT2018hco  ZTF18abxftqm  SansaStark_LRIS.mrt       SansaStark_LRIS.ascii
    AT2020ddv  ZTF20aamqmfk  Shae_20200609_DCT_v2.mrt  Shae_20200609_DCT_v2.ascii
    AT2020riz  ZTF20abrnwfc  Talisa_DCT.mrt            Talisa_DCT.ascii
    AT2019vcb  ZTF19acspeuw  Tormund_LRIS.mrt          Tormund_LRIS.ascii
    AT2019bhf  ZTF19aakswrb  Varys_DCT.mrt             Varys_DCT.ascii
    AT2020mbq  ZTF20abefeab  Yara_DBSP.mrt             Yara_DBSP.ascii

System requirements: The *.mrt files are formatted according to the machine
    readable format used by the AAS Journals and CDS/VizieR. Specific
    information on the structure of MRT files can be found at:

        AAS: https://journals.aas.org/mrt-overview/
        CDS: http://cds.u-strasbg.fr/doc/catstd.htx

    These files can be read in python using the astropy package:

        from astropy.table import Table
        data = Table.read("Yara_DBSP.mrt", format="ascii.cds")

    or with the most recent version of TOPCAT (> Version 4.8)
    
    The authors' original data files are supplied with extensions *ascii. 
    The are most common two column, space-delimited, data (27 files) and
    three LRIS plain-text "FITS" style spectra with eight columns of data.

Additional comments: 

================================================================================
