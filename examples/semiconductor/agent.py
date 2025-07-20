from __future__ import annotations

from argparse import ArgumentParser
from functools import cache

# pylint: disable=wrong-import-order
from data_and_knowledge import EXPERT_PROGRAMS, EXPERT_KNOWLEDGE
from openssa import DANA, ProgramStore, HTP, HTPlanner, OpenAILM
from semikong_lm import SemiKongLM
from gemini_lm import SemiconductorGeminiLM


@cache
def get_or_create_agent(use_semikong_lm: bool = True, use_gemini_lm: bool = False, max_depth=2, max_subtasks_per_decomp=4) -> DANA:
    if use_gemini_lm:
        lm = SemiconductorGeminiLM.from_defaults()
    elif use_semikong_lm:
        lm = SemiKongLM.from_defaults()
    else:
        lm = OpenAILM.from_defaults()

    program_store = ProgramStore(lm=lm)
    if EXPERT_PROGRAMS:
        from openssa.core.reasoning.ooda.ooda_reasoner import OodaReasoner
        reasoner = OodaReasoner(lm=lm)
        for program_name, htp_dict in EXPERT_PROGRAMS.items():
            htp = HTP.from_dict(htp_dict)
            # Recursively update all HTPs and sub-HTPs to use the correct LM
            def update_htp_reasoner(htp_instance):
                htp_instance.reasoner = reasoner
                for sub_htp in htp_instance.sub_htps:
                    update_htp_reasoner(sub_htp)
            update_htp_reasoner(htp)
            program_store.add_or_update_program(name=program_name, description=htp.task.ask, program=htp)

    return DANA(program_store=program_store,
                programmer=HTPlanner(lm=lm, max_depth=max_depth, max_subtasks_per_decomp=max_subtasks_per_decomp),
                knowledge={EXPERT_KNOWLEDGE} if EXPERT_KNOWLEDGE else None,
                resources={})


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('problem')
    args = arg_parser.parse_args()

    print(get_or_create_agent().solve(problem=args.problem))
