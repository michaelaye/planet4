#!/usr/bin/perl

=pod

=head1 NAME

HiBeautify

=head1 SUMMARY

HiBeautify creates special color products ("pretty pictures"); namely,
a synthetic RGB product and an enhanced IRB product that are cosmetically
improved over the raw color or normalized raw color. These special products
are extras associated with an eventual COLOR (IRB) RDR. They will be placed in 
the RDR Extras directory for the observation. This directory will be created if
needed.

It will typically be configured to use the HiColorNorm cubes as input and be
invoked after the HiColorNorm pipeline.

=head1 SYNOPSIS

HiBeautify
[-B<V>erbose] [-B<H>elp] [-B<Ca>talog E<lt>B<name>E<gt>]
[-B<Co>nfiguration E<lt>B<filename>E<gt>]
[-B<I>d E<lt>B<Conductor ID>E<gt>]
[-B<P>roduct_Version E<lt>NumberE<gt>]
PVL source file

=head1 OPTIONS

=over

=item   -B<Ca>talog E<lt>B<name>E<gt>

The name of the database catalog to use.

=item   -B<Co>nfiguration E<lt>B<filename>E<gt>

The Configuration file used to control some aspects of the cosmetic enhancement
and JP2/JPG creation.

=item   -B<I>d E<lt>B<Conductor ID>E<gt>

Provides the Conductor ID, to use in temporary files.
If unset, the process ID of the script will be used.

=item -B<P>roduct_Version E<lt>NumberE<gt>

To force the Extras to be created with a particular RDR version number, provide
it with the -Product_Version option.

If not specified, the Product_Version utility will be used to obtain the
next version number for the extras (defaulting to 1).

=item   -B<V>erbose

Show more messages and echo ISIS commands before they are run.

=item   -B<H>elp

Prints the command usage help.

=back

=head1 DESCRIPTION

HiBeautify uses the left & right HiColorNorm cubes (slithered, normalized,
three-band IRB cubes). It creates a mosaic of these which is saved as the _IRB
cub. This is used as the input source for subsequent steps.

HiBeautify consists of the following steps.

=cut

# namespace
package HiRISE::HiColor;

# modules etc.
use strict;
use warnings;
use English;
use File::Spec;
use Getopt::Long;
use POSIX 'ceil';

# base directory of HiRISE perl modules
use lib $ENV{HiRISE_ROOT} || '/HiRISE';

# more imports
use HiRISE::HiColor::HiColor qw(Run_Isis Check_Folder Location);
use HiRISE::HiColor::FrostStats;
use HiRISE::HiArch::Archiver;
use HiRISE::HiArch::HiArch qw(Get_Bands Get_MinMax);
use HiRISE::libHiRISE::ParsePVL;
use Pod::Usage;

# Revision
my $CVS_ID = 'HiBeautify 1.66 2010/12/22 22:55:05';

my $Command_Name = (File::Spec->splitpath($PROGRAM_NAME))[-1];
print STDERR "$Command_Name - $CVS_ID\n";

# defaults:
my $Id              = $PROCESS_ID;
my $Verbose         = 0;
my $Help            = 0;
my $Catalog        = 'HiRISE_Test';
my $Configuration   = "$HiRISE_ROOT/Configuration/HiBeautify/HiBeautify.conf";
my $Extras_Map      = "$HiRISE_ROOT/Configuration/HiBeautify/PVL_to_DB.Extras";
my $Product_Version = '';

my $status = GetOptions
(
    "Id=s"                  => \$Id,             # string
    "Help"                  => \$Help,           # flag
    "Verbose"               => \$Verbose,        # flag
    "Configuration=s"       => \$Configuration,  # string
    "Catalog=s"             => \$Catalog,        # string,
    "Product_Version=i"     => \$Product_Version # integer
);

pod2usage
(
    -verbose => 0,
    -exitval => ERR_SYNTAX
)
unless $status;

pod2usage
(
    -verbose => 2,
    -exitval => ERR_SYNTAX
)
if $Help;

pod2usage
(
    -verbose => 0,
    -message => "$Command_Name: At least one filename must be provided",
    -exitval => ERR_SYNTAX
)
unless @ARGV;

# input PVL file
my $Source = $ARGV[0];

unless ( -f $Source )
{
    print STDERR "Source file $Source could not be found.";
    exit ERR_INPUT;
}

# make sure path to configuration is absolute
$Configuration = File::Spec->rel2abs($Configuration);

# filename part of the source
my $Source_Filename = (File::Spec->splitpath($Source))[-1];

# obtain input product IDs
my @Halves = ParseArray(PvlGetKeyval($Source, "/HiBeautify/Product_Id"));

# obtain observation id from filename of source
my ($Observation_ID) = ($Source_Filename =~ /^(\w+)/);

print STDERR "Beginning HiBeautify for observation $Observation_ID\n";
print STDERR "Using configuration file $Configuration...\n\n";

# obtain the relative path for the ID
my $Location = Location($Observation_ID);

my
(
    $HiStitch_Folder,
    $HiJitReg_Folder,
    $HiSlither_Folder,
    $HiColorNorm_Folder,
    $HiBeautify_Folder,
    $HiColor_Extras
) = Check_Folder
(
    # HiStitch directory
    "$HiStitch_Data/$Location",

    # HiJitReg directory
    "$HiJitReg_Data/$Location",

    # HiColorNorm products directory
    "$HiSlither_Data/$Location",

    # HiColorNorm products directory
    "$HiColorNorm_Data/$Location",

    # working directory
    "$HiBeautify_Data/$Location",

    # Location of HiColor files
    "$HiRISE_ROOT/Data/Extras/RDR/$Location"
);

# change to our working directory
unless (chdir $HiBeautify_Folder)
{
    print STDERR "Unable to change to work folder $HiBeautify_Folder\n";
    exit ERR_INTERNAL;
}

# observation's HiJitReg input
my $hijitter_pvl  = "$HiJitReg_Folder/$Observation_ID.hijitreg.pvl";

# temp files to be created
my $irbmerged_cub = "$Temp_Folder/$Observation_ID" . "_IRB.$Id.cub";
my $rgbsynthb_cub = "$Temp_Folder/$Observation_ID" . "_B.$Id.cub";
my $brescaled_cub = "$Temp_Folder/$Observation_ID" . "_B.rescaled.$Id.cub";
my $rgbcubeit_txt = "$Temp_Folder/$Observation_ID" . "_RGB.$Id.txt";
my $rgbmerged_cub = "$Temp_Folder/$Observation_ID" . "_RGB.$Id.cub";
my $stretchR_cub  = "$Temp_Folder/$Observation_ID" . "_RED.stretch.$Id.cub";
my $stretchG_cub  = "$Temp_Folder/$Observation_ID" . "_BG.stretch.$Id.cub";
my $stretchB_cub  = "$Temp_Folder/$Observation_ID" . "_B.stretch.$Id.cub";

# holds meta-info
my %Image = ();

# things to look up from the config file
my @keywords =
qw(
    Synthetic_A_Coefficient
    Synthetic_B_Coefficient
    COLOR_IRB_Options
    COLOR_RGB_Options
    COLOR_JP2_Options
    Stretch_Reduction_Factor
    RGB_G_Gain_Coefficient
    RGB_B_Gain_Coefficient
    IRB_BG_Gain_Coefficient
    IRB_RED_Gain_Coefficient
);

# obtain configuration parameters
foreach my $keyword (@keywords)
{
   $Image{$keyword} = PvlGetKeyval($Configuration, "/Beautify/$keyword");
   
    unless (defined $Image{$keyword})
    {
        print STDERR "$keyword not found, cannot continue\n";
        
        exit ERR_INPUT;
    }   
    
    print STDERR qq($keyword = "$Image{$keyword}"\n) if $Verbose;
}

foreach my $keyword (qw(RGB_Minimum_Percent RGB_Maximum_Percent Ratio_Reduce))
{
   $Image{$keyword} = PvlGetKeyval($Configuration, "/HiMosMerge/HiMosMerge_$keyword");
   
    unless (defined $Image{$keyword})
    {
        print STDERR "$keyword not found, cannot continue\n";
        
        exit ERR_INPUT;
    }   
    
    print STDERR qq($keyword = "$Image{$keyword}"\n) if $Verbose;
}

# get parameters needed for mosaicking from the hijitreg source
foreach my $keyword (qw(Lines Samples Total_Width Image_Midpoint))
{
   $Image{$keyword} = PvlGetKeyval($hijitter_pvl, "/Image_Size/$keyword");
   
    unless (defined $Image{$keyword})
    {
        print STDERR "$keyword not found, cannot continue\n";
        
        exit ERR_INPUT;
    }  
    
    print STDERR qq($keyword = "$Image{$keyword}"\n) if $Verbose;    
}

=pod

Decompress the HiColorNorm cubes if necessary

=cut

foreach my $Half (@Halves)
{
    # decompress if needed
    # make sure we have the files we'll need
    if ( -s "$HiColorNorm_Folder/$Half.cub.gz" and !(-e "$HiColorNorm_Folder/$Half.cub"))
    {
        print STDERR "Found compressed $Half.cub.gz in $HiColorNorm_Folder\n";
        my $cmd = qq(gunzip -v "$HiColorNorm_Folder/$Half.cub.gz");
        print "$cmd\n";
        print qx($cmd);
        
        if ($CHILD_ERROR)
        {
            print STDERR "Decompression of $Half.cub.gz failed\n";
            exit ERR_EXTERNAL;
        }                 
    }
}

=pod

Create an IRB mosaic from the HiColorNorm halves.

If a half is missing, we create a mosaic with the proper width and place
the half in it at the proper location.

=cut

print STDERR "Creating color mosaic\n" if $Verbose;

if (scalar @Halves == 0)
{
    print STDERR "No products listed in input PVL $Source\n";
    exit ERR_INPUT;
}
elsif (scalar @Halves == 1)
{
    print STDERR "Warning, missing one half!\n" if $Verbose;

    # options for handmos
    my %options =
    (
         'FROM'    => "$HiColorNorm_Folder/$Halves[0].cub",
         'MOSAIC'  => $irbmerged_cub,
         'OUTLINE' => 1,
         'OUTBAND' => 1,
         'CREATE'  => 'Y',
         'NLINES'  => $Image{Lines},
         'NSAMP'   => $Image{Total_Width},
         'NBANDS'  => 3
    );

    if ($Halves[0] =~ /4$/)
    {
        $options{OUTSAMPLE} = 1;
    }
    else
    {
        $options{OUTSAMPLE} = $Image{Image_Midpoint};
    }

    # run handmos
    Run_Isis('handmos', \%options, $Verbose);

}
else
{
    print STDERR "Using both halves\n" if $Verbose;

    foreach my $N (4, 5)
    {
       # options for handmos
       my %options =
       (
         'FROM'    => "$HiColorNorm_Folder/$Observation_ID" . "_COLOR$N.cub",
         'MOSAIC'  => $irbmerged_cub,
         'OUTLINE' => 1,
         'OUTBAND' => 1
       );

       # only do these options for the first mosaic-ing
       if ($N == 4)
       {
          $options{CREATE} = 'Y';
          $options{OUTSAMPLE} = 1;
          $options{NLINES} = $Image{Lines};
          $options{NSAMP}  = $Image{Total_Width};
          $options{NBANDS} = 3;
       }
       # only do these options for the second mosaic-ing
       else
       {
          $options{OUTSAMPLE} = $Image{Image_Midpoint};
       }

       # run handmos
       Run_Isis('handmos', \%options, $Verbose);
    }
}

=pod

Subtract the unaltered RED band from the high pass filtered BG for synthetic
blue.

=cut

print STDERR "Creating synthetic B, subtracting RED from BG\n" if $Verbose;

Run_Isis
(
    'algebra',
    {
        'OP'    => 'subtract',
        'FROM1' => "$irbmerged_cub+3",
        'FROM2' => "$irbmerged_cub+2",
        'TO'    =>  $rgbsynthb_cub,
        'A'     => $Image{Synthetic_A_Coefficient},
        'B'     => $Image{Synthetic_B_Coefficient}
    },
    $Verbose
);

=pod

Determine the min and max DN values of each band (RED, IR, BG, B) we're working
with.

=cut

HiRISE::HiColor::Run_Isis
(
   'reduce',
   {
       'FROM'           => $rgbsynthb_cub,
       'TO'             => $brescaled_cub,
       'SSCALE'         => $Image{Ratio_Reduce},
       'LSCALE'         => $Image{Ratio_Reduce},
       'REDUCTION_TYPE' => 'scale'
   },
   $Verbose
);

($Image{B_MINIMUM}, $Image{B_MAXIMUM}, $Image{B_AVERAGE}) = HiRISE::HiArch::Get_MinMax
(
    $brescaled_cub, 
    1,     
    $Image{RGB_Minimum_Percent},
    $Image{RGB_Maximum_Percent},
    0, 
    $Id
 );

unless ( unlink $brescaled_cub )
{
    print STDERR "Could not remove temp file $brescaled_cub\n";
    exit ERR_EXTERNAL;
}

=pod

Determine if Frost/ICE may be present using FrostStats module.

=cut

my $frosty = HiRISE::HiColor::FrostStats->new($Catalog, $Configuration, $Verbose);
$frosty->init($Observation_ID);

my %frost = $frosty->findFrostStats($irbmerged_cub);

if ($Verbose)
{
    print STDERR "Frost Statistics\n";
    
    map { print STDERR "$_ : $frost{$_}\n"; } sort keys %frost;
}

print STDERR "Saving frost stats to database\n" if $Verbose;

#$frosty->saveFrostStats(\%frost);

map { $Image{$_} = $frost{$_} } keys %frost;

# Corresponding RDR product ID
my $Product_ID = $Observation_ID . "_COLOR";

print STDERR "Initializing archiver for $Product_ID\n";

# create instance of archiver for RDR Extras
my $Archiver = new HiRISE::HiArch::Archiver('Extras/RDR');

# share our verbosity setting
$Archiver->setIsVerbose($Verbose);

# we always overwrite existing extras
# Archiver's use of the Product_Version script will ensure
# that these are never PDS released files.
$Archiver->setWillClobber(1);

# use our config in case it isn't the default
$Archiver->setConfiguration($Configuration);

# use our catalog too
$Archiver->setCatalog($Catalog);

# specify the version to use if given an override (forced version number)
$Archiver->setNextVersion($Product_Version) if ($Product_Version);

# initialize with RDR product ID
$Archiver->init($Product_ID);

# RDR version that we are creating extras for
# (note, unless reprocessing, it won't have been created yet)
my $Version_ID = $Archiver->getNextVersion();

# make sure we have a version number
unless ($Version_ID)
{
    print STDERR "Could not determine version number for $Product_ID\n";
    exit ERR_EXTERNAL;
}

=pod

Tag if Frost/Ice was detected

=cut

my $cmd = "Tagger -Catalog $Catalog ";
$cmd .= "-Configuration $HiRISE_ROOT/Configuration/HiVali/tags.conf ";
$cmd .= "-Product $Product_ID#$Version_ID -Tag 'Frost or Ice' ";
    
if ($frost{FROST_FOUND} eq 'YES')
{
    $cmd .= "-Comment '$CVS_ID'";    
}
else
{
    $cmd .= "-Delete ";    
}

print "$cmd\n";

print qx($cmd);

if ($CHILD_ERROR)
{
    print STDERR "Warning, unable to set tag color failure!\n";
    exit ERR_EXTERNAL;
}

=pod

Stretch it out

=cut

unless ($Image{FROST_FOUND}  eq 'YES') 
{
    print STDERR "Frost/Ice was not detected, will boost BG and B color\n" if $Verbose;
    
    $Image{BG_MAXIMUM} = ($Image{BG_MAXIMUM} - $Image{BG_MINIMUM}) * $Image{RGB_G_Gain_Coefficient} + $Image{BG_MINIMUM};
    $Image{B_MAXIMUM} = ($Image{B_MAXIMUM} - $Image{B_MINIMUM}) * $Image{RGB_B_Gain_Coefficient} + $Image{B_MINIMUM};
}

# data to be filled in for PVL
my %Extras = 
(
    'RGB.NOMAP.JP2' =>
    {
        'STRETCH_MINIMA' => [ $Image{RED_MINIMUM}, $Image{BG_MINIMUM}, $Image{B_MINIMUM} ],
        'STRETCH_MAXIMA' => [ $Image{RED_MAXIMUM}, $Image{BG_MAXIMUM}, $Image{B_MAXIMUM} ]
    },
    'IRB.NOMAP.JP2' =>
    {
        'STRETCH_MINIMA' => [ $Image{IR_MINIMUM}, $Image{RED_MINIMUM}, $Image{BG_MINIMUM} ],
        'STRETCH_MAXIMA' => [ $Image{IR_MAXIMUM}, $Image{RED_MAXIMUM}, $Image{BG_MAXIMUM} ]
    }
    
);

=pod

Create an RGB cube using the RED from the IRB mosaic, the BG from the IRB mosaic
and the synthetic B that we just made.

=cut

my @RGB =
(
   "$irbmerged_cub+2", # RED
   "$irbmerged_cub+3", # BG
   $rgbsynthb_cub      # B
);

# write cubeit list file
open LIST, "> $rgbcubeit_txt" or do
{
        print STDERR "Could not write cubeit list RGB file\n";
        exit ERR_OUTPUT;
};

print LIST join "\n", @RGB;
print LIST "\n";

close LIST;

#  run cubeit (hicubeit does not handle +N syntax)
Run_Isis
(
    'cubeit',
    {
        'LIST' => $rgbcubeit_txt,
        'TO'   => $rgbmerged_cub
    },
    $Verbose
);


# remove the blue cube
unless ( unlink @RGB ) # == 3?
{
   print STDERR "Could not remove individual RGB files\n";
   exit ERR_EXTERNAL;
}

# Remove the list file
unless ( unlink $rgbcubeit_txt )
{
    print STDERR "Could not remove list file $rgbcubeit_txt\n";
    exit ERR_EXTERNAL;
}

=pod

Create the RGB.NOMAP and COLOR.NOMAP (IRB) Extras (JP2, browse and thumb).
Archive them into the proper versioned directories.

=cut

# do the IRB extras
&Create_JPEGs('IRB');

# do the RGB extras
&Create_JPEGs('RGB');

=pod

Clean up as configured

=cut

# remove, compress or preserve the HiSlither cubes
Clean_Up($Configuration, 'HiBeautify_clean_COLOR', $HiSlither_Folder . "/*_COLOR?.cub", $Verbose);

# remove, compress or preserve the HiStitch balance cubes
Clean_Up($Configuration, 'HiBeautify_clean_BAL', $HiStitch_Folder . "/*balance.cub", $Verbose);

=pod

Create PVL for Extras product mapping into HiCat

=cut

# current time
my $spacer = ",\n" . " " x 34;

my $IRB_Stretch_Minima = join $spacer, @{$Extras{'IRB.NOMAP.JP2'}{STRETCH_MINIMA}};
my $IRB_Stretch_Maxima = join $spacer, @{$Extras{'IRB.NOMAP.JP2'}{STRETCH_MAXIMA}};

my $RGB_Stretch_Minima = join $spacer, @{$Extras{'RGB.NOMAP.JP2'}{STRETCH_MINIMA}};
my $RGB_Stretch_Maxima = join $spacer, @{$Extras{'RGB.NOMAP.JP2'}{STRETCH_MAXIMA}};


# write our output, which will be the input for HiJitReg
open PVL, "> $Observation_ID.extras.pvl" or do
{
   print STDERR "Could not write to $Observation_ID.extras.pvl\n";
   exit ERR_OUTPUT;
};

my $Creation_Date = Now();
$Creation_Date =~ s/T/ /o;


print PVL<<"__PVL__"
/*
    HiBeautify Extras for $Observation_ID
    Created by $CVS_ID
    Created at $Creation_Date
*/
Group                          = Extras

   Creation_Time               = "$Creation_Date"
   
   Group                       = IRB

      Object                   = NOMAP

         PATHNAME              = $Extras{'IRB.NOMAP.JP2'}{PATHNAME}
         IMAGE_LINES           = $Extras{'IRB.NOMAP.JP2'}{IMAGE_LINES}
         LINE_SAMPLES          = $Extras{'IRB.NOMAP.JP2'}{LINE_SAMPLES}
         FILE_SIZE             = $Extras{'IRB.NOMAP.JP2'}{FILE_SIZE}

         STRETCH_MINIMA        = (
                                    $IRB_Stretch_Minima
                                 )
         STRETCH_MAXIMA        = (
                                    $IRB_Stretch_Maxima
                                 )

      End_Object

      Object                   = NOMAP_browse

         PATHNAME              = $Extras{'IRB.NOMAP.browse.jpg'}{PATHNAME}
         IMAGE_LINES           = $Extras{'IRB.NOMAP.browse.jpg'}{IMAGE_LINES}
         LINE_SAMPLES          = $Extras{'IRB.NOMAP.browse.jpg'}{LINE_SAMPLES}
         FILE_SIZE             = $Extras{'IRB.NOMAP.browse.jpg'}{FILE_SIZE}

      End_Object

      Object                   = NOMAP_thumb

         PATHNAME              = $Extras{'IRB.NOMAP.thumb.jpg'}{PATHNAME}
         IMAGE_LINES           = $Extras{'IRB.NOMAP.thumb.jpg'}{IMAGE_LINES}
         LINE_SAMPLES          = $Extras{'IRB.NOMAP.thumb.jpg'}{LINE_SAMPLES}
         FILE_SIZE             = $Extras{'IRB.NOMAP.thumb.jpg'}{FILE_SIZE}

      End_Object

   End_Group
   
   Group                       = RGB
   
      Object                   = NOMAP

         PATHNAME              = $Extras{'RGB.NOMAP.JP2'}{PATHNAME}
         IMAGE_LINES           = $Extras{'RGB.NOMAP.JP2'}{IMAGE_LINES}
         LINE_SAMPLES          = $Extras{'RGB.NOMAP.JP2'}{LINE_SAMPLES}
         FILE_SIZE             = $Extras{'RGB.NOMAP.JP2'}{FILE_SIZE}

         STRETCH_MINIMA        = (
                                    $RGB_Stretch_Minima
                                 )
         STRETCH_MAXIMA        = (
                                    $RGB_Stretch_Maxima
                                 )

      End_Object

      Object                   = NOMAP_browse

         PATHNAME              = $Extras{'RGB.NOMAP.browse.jpg'}{PATHNAME}
         IMAGE_LINES           = $Extras{'RGB.NOMAP.browse.jpg'}{IMAGE_LINES}
         LINE_SAMPLES          = $Extras{'RGB.NOMAP.browse.jpg'}{LINE_SAMPLES}
         FILE_SIZE             = $Extras{'RGB.NOMAP.browse.jpg'}{FILE_SIZE}

      End_Object

      Object                   = NOMAP_thumb

         PATHNAME              = $Extras{'RGB.NOMAP.thumb.jpg'}{PATHNAME}
         IMAGE_LINES           = $Extras{'RGB.NOMAP.thumb.jpg'}{IMAGE_LINES}
         LINE_SAMPLES          = $Extras{'RGB.NOMAP.thumb.jpg'}{LINE_SAMPLES}
         FILE_SIZE             = $Extras{'RGB.NOMAP.thumb.jpg'}{FILE_SIZE}

      End_Object

   End_Group

   Object                      = Frost_Stats
   
        FROST_FOUND                  = $Image{FROST_FOUND}        
        INCIDENCE_ANGLE_USED         = $Image{INCIDENCE_ANGLE_USED}
        REDUCE_FACTOR                = $Image{REDUCE_FACTOR}
        BG_PHOTO_CORRECTED           = $Image{BG_PHOTO_CORRECTED}
        BG_RED_RATIO_THRESHOLD       = $Image{BG_RED_RATIO_THRESHOLD}
        BG_PHOTO_CORRECTED_THRESHOLD = $Image{BG_PHOTO_CORRECTED_THRESHOLD}
        INCIDENCE_ANGLE_THRESHOLD    = $Image{INCIDENCE_ANGLE_THRESHOLD}
        BG_RED_RATIO_MASK_MINIMUM    = $Image{BG_RED_RATIO_MASK_MINIMUM}
        BG_RED_RATIO_MASK_MAXIMUM    = $Image{BG_RED_RATIO_MASK_MAXIMUM}
        BG_RED_RATIO_MASK_AVERAGE    = $Image{BG_RED_RATIO_MASK_AVERAGE}
        IR_RED_RATIO_MASK_MINIMUM    = $Image{IR_RED_RATIO_MASK_MINIMUM}
        IR_RED_RATIO_MASK_MAXIMUM    = $Image{IR_RED_RATIO_MASK_MAXIMUM}
        IR_RED_RATIO_MASK_AVERAGE    = $Image{IR_RED_RATIO_MASK_AVERAGE}
        BG_RED_RATIO_MINIMUM         = $Image{BG_RED_RATIO_MINIMUM}
        BG_RED_RATIO_MAXIMUM         = $Image{BG_RED_RATIO_MAXIMUM}
        BG_RED_RATIO_AVERAGE         = $Image{BG_RED_RATIO_AVERAGE}
        IR_RED_RATIO_MINIMUM         = $Image{IR_RED_RATIO_MINIMUM}
        IR_RED_RATIO_MAXIMUM         = $Image{IR_RED_RATIO_MAXIMUM}
        IR_RED_RATIO_AVERAGE         = $Image{IR_RED_RATIO_AVERAGE}
        IR_MINIMUM                   = $Image{IR_MINIMUM}
        IR_MAXIMUM                   = $Image{IR_MAXIMUM}
        IR_AVERAGE                   = $Image{IR_AVERAGE}
        RED_MINIMUM                  = $Image{RED_MINIMUM}
        RED_MAXIMUM                  = $Image{RED_MAXIMUM}
        RED_AVERAGE                  = $Image{RED_AVERAGE}
        BG_MINIMUM                   = $Image{BG_MINIMUM}
        BG_MAXIMUM                   = $Image{BG_MAXIMUM}
        BG_AVERAGE                   = $Image{BG_AVERAGE}  
   
   End_Object
   
End_Group

__PVL__
;

=pod

Use PVL_to_DB to update (or insert) the Extras info into HiCat.
Note, PVL_to_DB does not have a catalog option.

=cut

$cmd = "PVL_to_DB -Configuration $Configuration ";
$cmd .= "-Set OBSERVATION_ID=$Observation_ID ";
$cmd .= "-Set PRODUCT_ID=$Product_ID ";
$cmd .= "-Set VERSION_ID=$Version_ID ";
$cmd .= "-Verbose " if $Verbose;
$cmd .= "-Map $Extras_Map ";
$cmd .= "$Observation_ID.extras.pvl";

print "$cmd\n";

print qx($cmd);

if ($CHILD_ERROR)
{
    print STDERR "Unable to update HiCat for Extra Products\n";
    exit ERR_OUTPUT;
}


# done
exit SUCCESS;

=pod

=head2 Function Definitions

=head3 Create_JPEGs

Explode the stacked cube into separate band cubes.

Reduce each by the configured scale factor (adding the bin ratio if BG or IR).

Obtain the min and max DN from the reduced band cube.

Create the JP2 by converting the color mosaic with Isis2jp2, passing the min
and max DN from each reduced band cube.

Create JPG browse & thumb by converting the color mosaic with Isis2jpeg
using the configured stretch options.

=cut

sub Create_JPEGs
{
   # IRB or RGB
   my $type = shift;

   # product ID
   my $product = $Observation_ID . "_" . $type;

   print STDERR "Creating NOMAP extras for $product\n" if $Verbose;

   my $input = $type eq 'IRB' ? $irbmerged_cub : $rgbmerged_cub;
      
   # hash keys
   my ($NOMAP, $browse, $thumb) =
   (
       $type . '.NOMAP.JP2',
       $type . '.NOMAP.browse.jpg',
       $type . '.NOMAP.thumb.jpg'
   );

   # pathname of JP2
   $Extras{$NOMAP}{PATHNAME} = "$Temp_Folder/$product.NOMAP.JP2";

   # pathname of browse
   $Extras{$browse}{PATHNAME} = "$Temp_Folder/$product.NOMAP.browse.jpg";

   # pathname of thumb
   $Extras{$thumb}{PATHNAME} = "$Temp_Folder/$product.NOMAP.thumb.jpg";

   my $mina = join ',', map { $_ eq 'NULL' ? 0 : $_; } @{$Extras{$NOMAP}{STRETCH_MINIMA}};
   my $maxa = join ',', map { $_ eq 'NULL' ? 1 : $_; } @{$Extras{$NOMAP}{STRETCH_MAXIMA}};

   print STDERR "Creating JP2\n";

   my $cmd = "Isis2jp2 -In $input -Out $Extras{$NOMAP}{PATHNAME} ";
   $cmd .= $Image{COLOR_JP2_Options};
   $cmd .= " -Minimum $mina";
   $cmd .= " -Maximum $maxa";
   $cmd .= " -DN -ID $Id";

   print "$cmd\n";
   my @output = qx($cmd);
   print @output;

   if ($CHILD_ERROR)
   {
        print STDERR "Unable to create $product JP2 in $Temp_Folder\n";
        exit ERR_OUTPUT;
   }
      
   # pixel width of JP2
   $Extras{$NOMAP}{LINE_SAMPLES} = $Image{Total_Width};

   # pixel height of JP2
   $Extras{$NOMAP}{IMAGE_LINES} = $Image{Lines};

   # pixel width of browse
   $Extras{$browse}{LINE_SAMPLES} = 512;

   # pixel height of browse
   $Extras{$browse}{IMAGE_LINES} = POSIX::ceil($Image{Lines} / ($Image{Total_Width} / 512));

   # check against line limit
   if ($Extras{$browse}{IMAGE_LINES} > 32000)
   {
       # scale width
       $Extras{$browse}{LINE_SAMPLES} = POSIX::ceil(512 * 32000 / $Extras{$browse}{IMAGE_LINES});
       
       # set height to limit
       $Extras{$browse}{IMAGE_LINES} = 32000;

       $Extras{$thumb}{IMAGE_LINES} = 16000;

       $Extras{$thumb}{LINE_SAMPLES} = POSIX::ceil($Extras{$browse}{LINE_SAMPLES}/2);
   }
   else
   {
   
       # pixel width of thumb
       $Extras{$thumb}{LINE_SAMPLES} = 128;

       # pixel height of thumb
       $Extras{$thumb}{IMAGE_LINES} = POSIX::ceil($Extras{$browse}{IMAGE_LINES} / 4);

   }
   
   print STDERR "Reducing $type to browse scale cube\n";

   my $Temp_CUB = "$Temp_Folder/$product.browse.$Id.cub";

   Run_Isis
   (
        'reduce',
        {
            'FROM'           => $input,
            'TO'             => $Temp_CUB,
#            'MODE'           => 'TOTAL',
            'ALGORITHM'      => 'AVERAGE',
            'ONS'            => $Extras{$browse}{LINE_SAMPLES},
            'ONL'            => $Extras{$browse}{IMAGE_LINES},
            'VALIDPER'       => 1,
            'VPER_REPLACE'   => 'nearest'
        },
        $Verbose
   );

   print STDERR "Creating browse JPG\n";

   $cmd = "Isis2jpeg -In $Temp_CUB -Out $Extras{$browse}{PATHNAME} ";
   $cmd .= $Image{"COLOR_" . $type . "_Options"};

   if ($type eq 'RGB')
   {
       $cmd .= " -Minimum $mina";
       $cmd .= " -Maximum $maxa";       
   }
   
   print "$cmd\n";
   print qx($cmd);

   if ($CHILD_ERROR)
   {
         print STDERR "Unable to create $product browse jpg in $Temp_Folder\n";
         exit ERR_OUTPUT;
   }

   print STDERR "Creating thumb JPG\n";

   # convert the browse into the thumb, scaling appropriately
   $cmd = "convert $Extras{$browse}{PATHNAME} ";
   $cmd .= "-geometry $Extras{$thumb}{LINE_SAMPLES}x ";
   $cmd .= $Extras{$thumb}{PATHNAME};
   print "$cmd\n";
   print qx($cmd);

   if ($CHILD_ERROR)
   {
         print STDERR "Unable to create $product thumb jpeg in $Temp_Folder\n";
         exit ERR_OUTPUT;
   }

   unless (unlink $input)
   {
        print STDERR "Unable to remove $input\n";
        exit ERR_OUTPUT;
   }

   unless (unlink $Temp_CUB)
   {
        print STDERR "Unable to remove $Temp_CUB\n";
        exit ERR_OUTPUT;
   }

   # check and archive all created extras products
   foreach my $ID ($NOMAP, $browse, $thumb)
   {

       # check all products prior to archiving
       if (! -e $Extras{$ID}{PATHNAME})
       {
           print STDERR "Missing output for $ID\n";
           exit ERR_OUTPUT;
       }

       # determine size in bytes
       $Extras{$ID}{FILE_SIZE} = -s $Extras{$ID}{PATHNAME};

       unless ($Extras{$ID}{FILE_SIZE})
       {
           print STDERR "Created empty output for $ID\n";
           exit ERR_OUTPUT;
       }

       print STDERR "Archiving $type extras for color RDR version $Version_ID\n";

       # archive the product
       my $Path = $Archiver->archive($Extras{$ID}{PATHNAME});

       # make sure archiving succeeded
       if ($Path)
       {
           # update the hash with the final path
           $Extras{$ID}{PATHNAME} = $Path;
       }
       else
       {
           print STDERR "Could not be archived\n";
           exit ERR_OUTPUT;
       }

   }

   print STDERR "Finished $type extras\n" if $Verbose;
}

=pod

=head1 Exit Status

Zero on success, see HiColor.pm for definitions of the other exit codes.

=head1 Author

Guy McArthur, UA/HiROC

=head1 Copyright

Copyright (C) 2007-2010 Arizona Board of Regents on behalf of the Planetary
Image Research Laboratory, Lunar and Planetary Laboratory at the
University of Arizona.

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License, version 2, as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

=head1 Version

1.66 2010/12/22 22:55:05

=cut

