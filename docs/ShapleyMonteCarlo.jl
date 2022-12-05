
using DocumentFunction

@doc """
$(DocumentFunction.documentfunction(get_U;
    location=false,
    maintext="
    Generate matrix of projection on permutahedron.
    ",
    argtext=Dict("d"=>"dimension of permutation.")))
""" get_U

@doc """
$(DocumentFunction.documentfunction(get_perms_members;
    location=false,
    maintext="
    Generate values of permutations normalised members.
    ",
    argtext=Dict("d"=>"dimension of permutation.")))
""" get_perms_members

@doc """
$(DocumentFunction.documentfunction(argsort;
    location=false,
    maintext="
    Find nearest permutation to given vector.
    ",
    argtext=Dict("x"=>"input float vector.",
                 "p"=>"normalised permutation members.")))
""" argsort

@doc """
$(DocumentFunction.documentfunction(get_random_permutation;
    location=false,
    maintext="
    Generate random permutation.
    ",
    argtext=Dict("U"=>"projection matrix.",
                 "p"=>"normalised permutation members.")))
""" get_random_permutation

@doc """
$(DocumentFunction.documentfunction(get_random_subset;
    location=false,
    maintext="
    Generate random subset.
    ",
    argtext=Dict("U"=>"projection matrix.",
                 "p"=>"normalised members of origin permutation.",
                 "i"=>"the last integer which will not be included to the subset of permutation.")))
""" get_random_subset

@doc """
$(DocumentFunction.documentfunction(generate_subsets;
    location=false,
    maintext="
    Generate vector of random unique non-empty subset.
    ",
    argtext=Dict("d"=>"dimenstion of the initial permutation.",
                 "k"=>"count of subsets.",
                 "i"=>"the last integer which will not be included to the subset of permutation")))
""" generate_subsets
