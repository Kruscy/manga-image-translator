#!/usr/bin/env python3
"""
update.py - Manga Image Translator frissito szkript
Hasznalat: python update.py
"""

import subprocess
import sys
import os


def run_git(args, capture=True, check=False):
    """Git parancs futtatasa. Visszaadja (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=capture,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=check,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except FileNotFoundError:
        return -1, "", "GIT_NOT_FOUND"
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else ""


def check_git_available():
    """Ellenorzi, hogy a git elerheto-e."""
    code, out, err = run_git(["--version"])
    if code == -1 or err == "GIT_NOT_FOUND":
        print("HIBA: A 'git' parancs nem talalhato.")
        print("  Telepitsd a Gitet: https://git-scm.com/download/win")
        print("  Telepites utan indítsd ujra a parancssort.")
        sys.exit(1)
    print(f"Git verzio: {out}")


def check_current_branch():
    """Ellenorzi, hogy a jelenlegi ag 'main'-e."""
    code, branch, err = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if code != 0:
        print(f"HIBA: Nem sikerult lekerdezni az aktualis agat. ({err})")
        sys.exit(1)
    print(f"Aktualis ag: {branch}")
    if branch != "main":
        print(f"\nFIGYELMEZTETES: Nem a 'main' agon vagy, hanem: '{branch}'")
        print("Ez a szkript a 'main' ag frissitesere lett tervezve.")
        valasz = input("Folytatod ezen az agon? [i/n]: ").strip().lower()
        if valasz != "i":
            print("Frissites megszakítva.")
            sys.exit(0)
    return branch


def show_current_commits(label="Jelenlegi legutobb 5 commit"):
    """Megjeleníti az utolso 5 commitot."""
    print(f"\n--- {label} ---")
    code, out, err = run_git(["log", "--oneline", "-5"])
    if code != 0:
        print(f"  (Nem sikerult lekerdezni a commitokat: {err})")
    else:
        if out:
            for sor in out.splitlines():
                print(f"  {sor}")
        else:
            print("  (Nincsenek commitok)")
    print()


def check_local_changes():
    """Ellenorzi, hogy vannak-e helyi valtozasok."""
    code, out, err = run_git(["status", "--porcelain"])
    if code != 0:
        return False
    return bool(out.strip())


def stash_changes():
    """Elrakja (stash) a helyi valtozasokat. Visszaadja True-t, ha volt mit elrakni."""
    print("\nFIGYELMEZTETES: Helyi, nem commitalt valtozasok talalthatoak.")
    print("Ezeket ideiglenesen elrakjuk (git stash), majd a frissites utan visszaallítjuk.")
    valasz = input("Folytatod? [i/n]: ").strip().lower()
    if valasz != "i":
        print("Frissites megszakítva.")
        sys.exit(0)

    print("\nHelyi valtozasok elrakasa (git stash)...")
    code, out, err = run_git(["stash", "push", "-m", "update.py: automatikus stash frissites elott"])
    if code != 0:
        print(f"HIBA: Nem sikerult a stash muvelet. ({err})")
        sys.exit(1)

    if "No local changes to save" in out or "No local changes to save" in err:
        print("  Nem volt mit elrakni.")
        return False

    print(f"  Elrakva: {out or 'OK'}")
    return True


def get_diff_files_before_pull():
    """Lekeri, hogy a pull utan mely fajlok valtoztak (a remote es a helyi HEAD kozott)."""
    # Fetch elott megnezzuk, mi valtozna
    code, out, err = run_git(["diff", "--name-only", "HEAD", "origin/main"])
    if code != 0:
        return []
    return out.splitlines()


def git_pull():
    """Lefuttatja a git pull origin main parancsot."""
    print("\nFrissites letoltese (git pull origin main)...")
    code, out, err = run_git(["pull", "origin", "main"])

    if code != 0:
        combined = (out + "\n" + err).lower()

        if "could not resolve host" in combined or "unable to connect" in combined or "network" in combined:
            print("HIBA: Halozati hiba - nem sikerult csatlakozni a GitHub-hoz.")
            print("  Ellenorizd az internetkapcsolatodat, majd probald ujra.")
            return False

        if "conflict" in combined or "merge conflict" in combined:
            print("HIBA: Merge konfliktus keletkezett!")
            print("  A konfliktusokat kezzel kell megoldani.")
            print("  Hasznald: git status  (a konfliktus fajlok megtekintesehez)")
            print("  Hasznald: git diff    (a valtozasok megtekintésehez)")
            return False

        if "authentication" in combined or "403" in combined or "permission denied" in combined:
            print("HIBA: Hozzaferesi hiba - nincs jogosultsagod a repot letolteni.")
            print("  Ellenorizd a GitHub bejelentkezési adataidat.")
            return False

        print(f"HIBA: git pull sikertelen (kod: {code})")
        if out:
            print(f"  stdout: {out}")
        if err:
            print(f"  stderr: {err}")
        return False

    print(f"  {out if out else 'Frissites sikeres.'}")
    if err and err.lower() not in ("", "already up to date."):
        print(f"  (git uzenet: {err})")

    return True


def pop_stash():
    """Visszaallítja a korábban elrakott valtozasokat."""
    print("\nHelyi valtozasok visszaallítasa (git stash pop)...")
    code, out, err = run_git(["stash", "pop"])
    if code != 0:
        print("FIGYELMEZTES: Nem sikerult automatikusan visszaallítani a helyi valtozasokat.")
        print("  Kezzel allíthato vissza: git stash pop")
        print(f"  Hiba: {err}")
    else:
        print(f"  Visszaallítva: {out or 'OK'}")


def suggest_pip_install(changed_files):
    """Ha a requirements.txt valtozot, javasol pip install-t."""
    req_files = [
        f for f in changed_files
        if "requirements" in f.lower() and f.endswith(".txt")
    ]
    if req_files:
        print("\nFIGYELMEZTETES: A kovetkezo fajlok megvaltoztak:")
        for f in req_files:
            print(f"  - {f}")
        print("\nAjanlott a fuggoségeket frissíteni:")
        print(f"  {sys.executable} -m pip install -r requirements.txt")
        valasz = input("\nMost futtatod a pip install -r requirements.txt parancsot? [i/n]: ").strip().lower()
        if valasz == "i":
            print("\nFuggoségek telepítése...")
            req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", req_path],
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode == 0:
                print("Fuggoségek sikeresen frissítve.")
            else:
                print("FIGYELMEZTES: A pip install nem futott le teljesen sikeresen.")
                print(f"  Futtasd kezzel: pip install -r requirements.txt")


def main():
    print("=" * 60)
    print("  Manga Image Translator - Frissíto Szkript")
    print("=" * 60)

    # 1. Git ellenorzes
    print("\n[1/8] Git ellenorzese...")
    check_git_available()

    # 2. Ag ellenorzese
    print("\n[2/8] Aktualis ag ellenorzese...")
    check_current_branch()

    # 3. Jelenlegi commitok megjelenítese
    print("\n[3/8] Jelenlegi commit elozmenyek:")
    show_current_commits("Frissítés ELOTTI legutobb 5 commit")

    # 4. Megerosites
    print("[4/8] Megerosites")
    valasz = input("Frissítesz? [i/n]: ").strip().lower()
    if valasz != "i":
        print("Frissites megszakítva.")
        sys.exit(0)

    # 5. Helyi valtozasok ellenorzese / stash
    print("\n[5/8] Helyi valtozasok ellenorzese...")
    van_valtozas = check_local_changes()
    stash_keszult = False
    if van_valtozas:
        stash_keszult = stash_changes()
    else:
        print("  Nincsenek helyi, nem commitalt valtozasok.")

    # 6. Diff lekerese a pull elott (fetch utan)
    print("\n[6/8] Valtozasok ellenorzese a szerveren...")
    run_git(["fetch", "origin", "main"])
    valtozot_fajlok = get_diff_files_before_pull()
    if valtozot_fajlok:
        print(f"  {len(valtozot_fajlok)} fajl fog valtozni a frissites soran.")
    else:
        print("  Nincsenek uj valtozasok a szerveren (mar naprakesz vagy).")

    # 7. git pull
    print("\n[7/8] Frissítés...")
    siker = git_pull()
    if not siker:
        if stash_keszult:
            print("\nA git pull sikertelen volt. Megprobaljuk visszaallítani a helyi valtozasokat...")
            pop_stash()
        sys.exit(1)

    # 8. requirements.txt javaslat
    if valtozot_fajlok:
        suggest_pip_install(valtozot_fajlok)

    # 9. Stash visszaallítasa
    if stash_keszult:
        pop_stash()

    # 10. Uj commitok megjelenítese
    print()
    show_current_commits("Frissítés UTANI legutobb 5 commit")

    print("=" * 60)
    print("  Frissítés befejezve!")
    print("=" * 60)


if __name__ == "__main__":
    main()
