#
#//  main.py
#//  rangerKnockoffPy
#//
#//  Created by Evan Mason on 2/10/26.
#//

from rangerknockoffpy import forestKnockoff

from os import path

def main() -> None:
    rangerKnockoffR_path: str = path.join(
        path.dirname( __file__ ),
        'src', 'rangerKnockoff',
    )

    rController: forestKnockoff.RController = forestKnockoff.RController(
        package_dir = rangerKnockoffR_path,
        force_install = True,
    )
    
    print("rangerknockoffpy/main.py DONE")
    
    return
#/def main

if __name__ == "__main__":
    main()
#

