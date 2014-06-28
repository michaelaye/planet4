
=pod

This file contains common routines and definitions used in HiColor scripts.

=cut

# namespace
package HiRISE::HiColor;

# modules etc.
use strict;
use warnings;
use English;
use POSIX 'strftime';
use File::Spec;

# we are a module
require Exporter;

# these variables have file scope
use vars qw
(
    $CVS_ID $HiRISE_ROOT $Data_Location $HiColorNorm_Data $HiBeautify_Data
    $HiStitch_Data $HiSlither_Data $HiJitReg_Data $HiColorInit_Data $HiGeomInit_Data
    $Temp_Folder @ISA @EXPORT @EXPORT_OK %CCD_Correspondence
);

# our subroutines
use subs qw(Run_Isis);

# Path to HiRISE (testing or production environment)
$HiRISE_ROOT = $ENV{'HiRISE_ROOT'} || '/HiRISE';

# Location of HiStitch files
$HiStitch_Data = "$HiRISE_ROOT/Data/HiStitch";

# Location of HiColorInit files
$HiColorInit_Data = "$HiRISE_ROOT/Data/HiColorInit";

# Location of HiJitReg files
$HiJitReg_Data = "$HiRISE_ROOT/Data/HiJitReg";

# Location of HiSlither files
$HiSlither_Data = "$HiRISE_ROOT/Data/HiSlither";

# Location of HiColorNorm files
$HiColorNorm_Data = "$HiRISE_ROOT/Data/HiColorNorm";

# Location of HiBeautify files
$HiBeautify_Data = "$HiRISE_ROOT/Data/HiBeautify";

# Location of HiGeomInit files
$HiGeomInit_Data = "$HiRISE_ROOT/Data/HiGeomInit";

# where any tmp files should go
$Temp_Folder = '/tmp';

# return codes
use constant ERR_SYNTAX   => -1;
use constant SUCCESS      =>  0;
use constant ERR_INPUT    =>  1;
use constant ERR_ISIS     =>  2;
use constant ERR_OUTPUT   =>  3;
use constant ERR_EXTERNAL =>  4;
use constant ERR_INTERNAL =>  5;

# ratio of color to red for enlargement, correction of optical distortion
use constant OPTICAL_ENLARGEMENT_RATIO => 1.0006;

# extend Exporter class
@ISA = 'Exporter';

# shared methods and variables
@EXPORT_OK = qw(Run_Isis Check_Folder Next_Pipeline Delta_Offset Clean_Up Median Now Location);

# variables which get exported automatically
@EXPORT = qw
(
    $HiRISE_ROOT $Temp_Folder OPTICAL_ENLARGEMENT_RATIO SUCCESS
    ERR_SYNTAX ERR_INPUT ERR_ISIS ERR_OUTPUT ERR_EXTERNAL ERR_INTERNAL
    $HiSlither_Data $HiJitReg_Data $HiColorInit_Data $HiStitch_Data
    $HiColorNorm_Data $HiBeautify_Data $HiGeomInit_Data %CCD_Correspondence
);

# Revision
$CVS_ID = 'HiColor.pm 1.33 2009/05/14 19:56:42';

# define color pairs
%CCD_Correspondence =
(
   'IR10' => 'RED4',
   'BG12' => 'RED4',
   'IR11' => 'RED5',
   'BG13' => 'RED5'
);

#
# Routine to run an ISIS command, exit on failure
# Returns the output of the command
# Parameters:
# a) command name, may include option
# b) [OPTIONAL] hash of optional parameter to value
# c) [OPTIONAL] verbosity level, 0 or 1 (echo command)
sub Run_Isis
{
    my $cmd = shift;
    my $ops = shift;
    my $vrb = shift;

    # if given hash append to the command KEY=VAL pairs
    map { $cmd .= " $_=" . $ops->{$_} } keys (%{$ops});

    # echo the command if verbose
    print "$cmd\n" if ($vrb);

    # capture output
    my @ret = qx($cmd);

    # check return code
    if ($CHILD_ERROR)
    {
        print STDERR "@ret\n" if ($vrb);
        print STDERR "Error running ISIS command: $cmd\n";
        print STDERR "Return code = $CHILD_ERROR\n";

        exit ERR_ISIS;
    }

    # the newline can cause problems, remove it
    chomp @ret;

    # return output
    return $ret[0];
}

#
# Routine to calculate delta offset in lines between red and color ccd
# Returns the delta in lines
# Parameters:
# a) Color binning mode (integer)
# b) Color TDI setting  (integer)
# a) RED binning mode (integer)
# b) RED TDI setting  (integer)
sub Delta_Offset
{
    my ($Color_Bin, $Color_TDI, $Red_Bin, $Red_TDI) = @_;

    my $offset = 200 * ($Color_Bin - $Red_Bin) + $Color_TDI - $Red_TDI;

    return $offset / $Red_Bin;
}

#
# Routine to make sure a directory exists, creating it if necessary
# Parameters: full or relative path to folder
sub Check_Folder
{
    foreach my $Folder (@_)
    {

        next if ( -e $Folder );

        qx(mkdir -p "$Folder");

        unless ( -e $Folder )
        {
            print STDERR "Unable to create folder $Folder\n";
            exit ERR_INTERNAL;
        }
    }

    return @_;
}

# Routine to clean up depending on configuration settings
# a) configuration file (full path)
# b) keyword (string)
# c) expression for file glob
sub Clean_Up
{
    my ($configuration, $keyword, $fileglob, $vrb) = @_;

    my $clean_option = Run_Isis
    (
        'getkey',
        {
                'FROM'    => $configuration,
                'GRPNAME' => 'Conductor',
                'KEYWORD' => $keyword
        },
        $vrb
    );

    if ($clean_option =~ /(remove)|(delete)|(clean)|(yes)|(true)/i)
    {
        print STDERR "Cleaning up $fileglob\n";
        my $count = unlink glob $fileglob;
        print STDERR "Cleaned up $count file(s).\n";
    }
    elsif ($clean_option =~ /(gzip)|(compress)/i)
    {

        print STDERR "Compressing $fileglob\n";
        my $status = system 'gzip', '-f', glob $fileglob;

        return if ($status == 0 and $CHILD_ERROR == 0);

        print STDERR "Failed to compress $keyword intermediate products\n";

        exit ERR_EXTERNAL;
    }

}

=pod

Submits a source to the next pipeline as driven by Configuration.

=cut

sub Next_Pipeline
{
    my ($catalog, $configuration, $current_pipe, $next_pipe, $source, $vrb, $sfx) = @_;

    # return unless given name of the next pipeline
    return unless ($next_pipe ne '');

    my $update = Run_Isis
    (
        'getkey',
        {
                'FROM'    => $configuration,
                'GRPNAME' => 'Conductor',
                'KEYWORD' => $current_pipe . "_update_pipeline"
        },
        $vrb
    );

    # return unless config tells us to update the next pipeline
    return unless ($update eq 'TRUE');

    # determine product id for tracking status
    (File::Spec->splitpath($source))[-1] =~ /^([^.]+)/;
    
    my $product_id = $1;
    $product_id .= "_$sfx" if $sfx;
    
    print STDERR "Sending $source to $next_pipe\n" if ($vrb);

    my $ret = qx
    (
        Next_Pipeline                      \\
             -Catalog $catalog             \\
             -Configuration $configuration \\
             -Pipeline $next_pipe          \\
             -Product $product_id          \\
             -Verbose                      \\
             $source
    );

    print STDERR $ret if ($vrb);

    # check return code
    if ($CHILD_ERROR)
    {

        print STDERR "Error running Next_Pipeline\n";
        print STDERR "Return code = $CHILD_ERROR\n";

        exit ERR_EXTERNAL;
    }
}

=pod

Median function, does a "mean median": for an odd-length array returns the middle
element, for an even-length array returns the mean of the two middle elements.

This should be equivilant to Excel's median function.

=cut

sub Median
{
    my $aref = shift;
    my @elements = sort {$a <=> $b} @$aref;

    if (@elements % 2)
    {
        return $elements[@elements/2];
    }
    else
    {
        return ($elements[@elements/2-1] + $elements[@elements/2]) / 2;
    }
}

=pod

Returns the current time in standard format.

=cut

sub Now
{
    return POSIX::strftime('%FT%T', localtime);
}

=pod

Runs Data_Location for an Observation_ID and returns the result.

=cut
sub Location
{
   my $Observation_ID = shift;

   # obtain the relative path for the ID
   my $loc = qx(Data_Location $Observation_ID);

   # check for error
   if ($CHILD_ERROR)
   {
      print STDERR "Data_Location failed for $Observation_ID\n";
      exit ERR_EXTERNAL;
   }

   # friggin newline
   chomp $loc;

   return $loc;
}

1;

